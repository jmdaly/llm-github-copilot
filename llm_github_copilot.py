import llm
import os
import json
import time
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator


@llm.hookimpl
def register_models(register):
    """Register all GitHub Copilot models with the LLM CLI tool."""
    # Create an authenticator to fetch models
    authenticator = GitHubCopilotAuthenticator()

    try:
        # Register the default model first
        default_model = GitHubCopilot()
        register(default_model)

        # Try to fetch available models
        models = fetch_available_models(authenticator)

        # Register all model variants
        for model_id in models:
            if model_id == default_model.model_id:
                continue  # Skip the default model as it's already registered

            model = GitHubCopilot()
            model.model_id = model_id
            register(model)

    except Exception as e:
        print(f"Warning: Failed to fetch GitHub Copilot models: {str(e)}")
        print("Falling back to default model only")


def fetch_available_models(authenticator):
    """
    Fetch available models from GitHub Copilot.

    Args:
        authenticator: The GitHubCopilotAuthenticator instance

    Returns:
        Set of model IDs
    """
    try:
        # Get API key
        api_key = authenticator.get_api_key()

        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "editor-version": "vscode/1.85.1",
        }

        # Make request to get models
        response = httpx.get(
            "https://api.githubcopilot.com/models", headers=headers, timeout=30
        )
        response.raise_for_status()

        # Parse response
        models_data = response.json()
        model_ids = set(["github-copilot"])  # Always include default model

        # Process models from response - models are in the "data" field
        for model in models_data.get("data", []):
            model_id = model.get("id")
            if model_id and model_id != "gpt-4o":  # Skip the default model ID
                model_ids.add(f"github-copilot/{model_id}")

        return model_ids

    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        # Return a minimal set of known models as fallback
        return {"github-copilot"}


class GitHubCopilotAuthenticator:
    """
    Handles authentication with GitHub Copilot using device code flow.
    """

    # GitHub API constants
    GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"  # GitHub Copilot client ID
    GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
    GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_API_KEY_URL = "https://api.github.com/copilot_internal/v2/token"

    # Default headers for GitHub API
    DEFAULT_HEADERS = {
        "accept": "application/json",
        "editor-version": "vscode/1.85.1",
        "accept-encoding": "gzip,deflate,br",
        "content-type": "application/json",
    }

    # Authentication constants
    MAX_LOGIN_ATTEMPTS = 3
    MAX_POLL_ATTEMPTS = 12
    POLL_INTERVAL = 5  # seconds

    def __init__(self) -> None:
        # Token storage paths
        self.token_dir = os.getenv(
            "GITHUB_COPILOT_TOKEN_DIR",
            os.path.expanduser("~/.config/llm/github_copilot"),
        )
        self.access_token_file = os.path.join(
            self.token_dir,
            os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token"),
        )
        self.api_key_file = os.path.join(
            self.token_dir, os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json")
        )
        self._ensure_token_dir()

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _get_github_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Generate standard GitHub headers for API requests."""
        headers = self.DEFAULT_HEADERS.copy()

        if access_token:
            headers["authorization"] = f"token {access_token}"

        return headers

    def get_access_token(self) -> str:
        """
        Get GitHub access token, refreshing if necessary.
        """
        # Try to read existing token
        try:
            with open(self.access_token_file, "r") as f:
                access_token = f.read().strip()
                if access_token:
                    return access_token
        except (IOError, FileNotFoundError):
            # File doesn't exist or can't be read
            pass

        # No valid token found, need to login
        for attempt in range(self.MAX_LOGIN_ATTEMPTS):
            try:
                access_token = self._login()
                # Save the new token
                try:
                    with open(self.access_token_file, "w") as f:
                        f.write(access_token)
                    os.chmod(self.access_token_file, 0o600)
                except (IOError, FileNotFoundError):
                    print("Error saving access token to file")
                return access_token
            except Exception as e:
                print(
                    f"Login attempt {attempt + 1}/{self.MAX_LOGIN_ATTEMPTS} failed: {str(e)}"
                )
                if attempt == self.MAX_LOGIN_ATTEMPTS - 1:  # Last attempt
                    raise Exception(
                        f"Failed to get access token after {self.MAX_LOGIN_ATTEMPTS} attempts"
                    )
                continue

    def get_api_key(self) -> str:
        """
        Get the API key, refreshing if necessary.
        """
        try:
            with open(self.api_key_file, "r") as f:
                api_key_info = json.load(f)
                if api_key_info.get("expires_at", 0) > datetime.now().timestamp():
                    return api_key_info.get("token")
        except (IOError, json.JSONDecodeError, KeyError):
            pass

        try:
            api_key_info = self._refresh_api_key()
            with open(self.api_key_file, "w") as f:
                json.dump(api_key_info, f)
                os.chmod(self.api_key_file, 0o600)
            return api_key_info.get("token")
        except Exception as e:
            raise Exception(f"Failed to get API key: {str(e)}")

    def _get_device_code(self) -> Dict[str, str]:
        """
        Get a device code for GitHub authentication.
        """
        required_fields = ["device_code", "user_code", "verification_uri"]

        try:
            client = httpx.Client()
            resp = client.post(
                self.GITHUB_DEVICE_CODE_URL,
                headers=self._get_github_headers(),
                json={"client_id": self.GITHUB_CLIENT_ID, "scope": "read:user"},
                timeout=30,
            )
            resp.raise_for_status()
            resp_json = resp.json()

            # Validate response contains required fields
            if not all(field in resp_json for field in required_fields):
                missing = [f for f in required_fields if f not in resp_json]
                raise Exception(
                    f"Response missing required fields: {', '.join(missing)}"
                )

            return resp_json
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Request error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to get device code: {str(e)}")

    def _poll_for_access_token(self, device_code: str) -> str:
        """
        Poll for an access token after user authentication.
        """
        client = httpx.Client()

        for attempt in range(self.MAX_POLL_ATTEMPTS):
            try:
                resp = client.post(
                    self.GITHUB_ACCESS_TOKEN_URL,
                    headers=self._get_github_headers(),
                    json={
                        "client_id": self.GITHUB_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                resp_json = resp.json()

                if "access_token" in resp_json:
                    print("Authentication successful!")
                    return resp_json["access_token"]
                elif (
                    "error" in resp_json
                    and resp_json.get("error") == "authorization_pending"
                ):
                    print(
                        f"Waiting for authorization... (attempt {attempt + 1}/{self.MAX_POLL_ATTEMPTS})"
                    )
                else:
                    error_msg = resp_json.get(
                        "error_description", resp_json.get("error", "Unknown error")
                    )
                    print(f"Unexpected response: {error_msg}")
            except httpx.HTTPStatusError as e:
                raise Exception(
                    f"HTTP error {e.response.status_code}: {e.response.text}"
                )
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")
            except Exception as e:
                raise Exception(f"Failed to get access token: {str(e)}")

            time.sleep(self.POLL_INTERVAL)

        raise Exception("Timed out waiting for user to authorize the device")

    def _login(self) -> str:
        """
        Login to GitHub Copilot using device code flow.
        """
        device_code_info = self._get_device_code()

        device_code = device_code_info["device_code"]
        user_code = device_code_info["user_code"]
        verification_uri = device_code_info["verification_uri"]

        print(
            f"\nPlease visit {verification_uri} and enter code {user_code} to authenticate GitHub Copilot.\n"
        )

        return self._poll_for_access_token(device_code)

    def _refresh_api_key(self) -> Dict[str, Any]:
        """
        Refresh the API key using the access token.
        """
        access_token = self.get_access_token()
        headers = self._get_github_headers(access_token)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = httpx.Client()
                response = client.get(
                    self.GITHUB_API_KEY_URL, headers=headers, timeout=30
                )
                response.raise_for_status()

                response_json = response.json()

                if "token" in response_json:
                    return response_json
                else:
                    print(f"API key response missing token: {response_json}")
            except httpx.HTTPStatusError as e:
                print(f"HTTP error {e.response.status_code}: {e.response.text}")
            except httpx.RequestError as e:
                print(f"Request error: {str(e)}")
            except Exception as e:
                print(
                    f"Error refreshing API key (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

            if attempt < max_retries - 1:
                time.sleep(1)

        raise Exception("Failed to refresh API key after maximum retries")


class GitHubCopilot(llm.Model):
    """
    GitHub Copilot model implementation for LLM.
    """

    model_id = "github-copilot"
    can_stream = True

    # API base URL
    API_BASE = "https://api.githubcopilot.com"

    # Default system message
    DEFAULT_SYSTEM_MESSAGE = "You are GitHub Copilot, an AI programming assistant."

    # Default request timeout in seconds
    DEFAULT_TIMEOUT = 120
    NON_STREAMING_TIMEOUT = 180

    # Default model mapping
    DEFAULT_MODEL_MAPPING = "gpt-4o"

    # Cache for model mappings
    _model_mappings = None

    # Cache for streaming models
    _streaming_models = None

    class Options(llm.Options):
        """
        Options for the GitHub Copilot model.
        """

        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate", default=None
        )

        temperature: Optional[float] = Field(
            description="Controls randomness in the output (0-1)",
            default=None,
        )

        @field_validator("max_tokens")
        def validate_max_tokens(cls, max_tokens):
            if max_tokens is None:
                return None
            if max_tokens < 1:
                raise ValueError("max_tokens must be >= 1")
            return max_tokens

        @field_validator("temperature")
        def validate_temperature(cls, temperature):
            if temperature is None:
                return None
            if not 0 <= temperature <= 1:
                raise ValueError("temperature must be between 0 and 1")
            return temperature

    def __init__(self):
        """Initialize the GitHub Copilot model."""
        self.authenticator = GitHubCopilotAuthenticator()

    @classmethod
    def get_model_mappings(cls):
        """
        Get model mappings, fetching them if not already cached.

        Returns:
            Dict mapping model IDs to API model names
        """
        if cls._model_mappings is None:
            try:
                # Create a temporary authenticator to fetch models
                authenticator = GitHubCopilotAuthenticator()
                api_key = authenticator.get_api_key()

                # Prepare headers
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "editor-version": "vscode/1.85.1",
                    "editor-plugin-version": "copilot/1.155.0",
                }

                # Make request to get models
                response = httpx.get(
                    "https://api.githubcopilot.com/models", headers=headers, timeout=30
                )
                response.raise_for_status()

                # Parse response
                models_data = response.json()
                mappings = {"github-copilot": cls.DEFAULT_MODEL_MAPPING}

                # Process models from response - models are in the "data" field
                for model in models_data.get("data", []):
                    model_id = model.get("id")
                    if model_id:
                        # Add all models, including the default one
                        mappings[f"github-copilot/{model_id}"] = model_id

                cls._model_mappings = mappings

            except Exception as e:
                print(f"Error fetching model mappings: {str(e)}")
                # Fallback to basic mappings
                cls._model_mappings = {
                    "github-copilot": "gpt-4o",
                }

        return cls._model_mappings

    @classmethod
    def get_streaming_models(cls):
        """
        Get list of models that support streaming.

        Returns:
            List of model names that support streaming
        """
        if cls._streaming_models is None:
            try:
                # Create a temporary authenticator to fetch models
                authenticator = GitHubCopilotAuthenticator()
                api_key = authenticator.get_api_key()

                # Prepare headers
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "editor-version": "vscode/1.85.1",
                }

                # Make request to get models
                response = httpx.get(
                    "https://api.githubcopilot.com/models", headers=headers, timeout=30
                )
                response.raise_for_status()

                # Parse response
                models_data = response.json()
                streaming_models = []

                # Process models from response - models are in the "data" field
                for model in models_data.get("data", []):
                    model_id = model.get("id")
                    # Check if model supports streaming
                    capabilities = model.get("capabilities", {})
                    supports = capabilities.get("supports", {})

                    if supports.get("streaming", False) and model_id:
                        streaming_models.append(model_id)

                # Always include default model
                if cls.DEFAULT_MODEL_MAPPING not in streaming_models:
                    streaming_models.append(cls.DEFAULT_MODEL_MAPPING)

                cls._streaming_models = streaming_models

            except Exception as e:
                print(f"Error fetching streaming models: {str(e)}")
                # Fallback to assuming all models support streaming
                mappings = cls.get_model_mappings()
                cls._streaming_models = list(mappings.values())

        return cls._streaming_models

    def _get_model_for_api(self, model: str) -> str:
        """
        Convert model name to API-compatible format.

        Args:
            model: The model identifier (e.g., "github-copilot/o1")

        Returns:
            The API model name (e.g., "o1")
        """
        # Get model mappings
        mappings = self.get_model_mappings()

        # Strip provider prefix if present
        if "/" in model:
            _, model_name = model.split("/", 1)
            if model_name in mappings.values():
                return model_name

        # Use the mapping or default to gpt-4o
        return mappings.get(model, self.DEFAULT_MODEL_MAPPING)

    def _non_streaming_request(self, prompt, headers, payload, model_name):
        """
        Handle non-streaming requests.

        Args:
            prompt: The user prompt
            headers: Request headers
            payload: Request payload
            model_name: The model name for logging

        Yields:
            Generated text content
        """
        try:
            # Ensure stream is set to false
            payload["stream"] = False

            api_response = httpx.post(
                f"{self.API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.NON_STREAMING_TIMEOUT,
            )
            api_response.raise_for_status()

            # Try to parse JSON
            try:
                json_data = api_response.json()

                if "choices" in json_data and json_data["choices"]:
                    choice = json_data["choices"][0]

                    # Handle different response formats
                    if "message" in choice and choice["message"]:
                        content = choice["message"].get("content", "")
                        if content:
                            yield content
                            return
                    elif "text" in choice:
                        content = choice.get("text", "")
                        if content:
                            yield content
                            return
                    elif "content" in choice:
                        content = choice.get("content", "")
                        if content:
                            yield content
                            return

                # If we couldn't extract content through known paths, try to find it elsewhere
                if "content" in json_data:
                    yield json_data["content"]
                    return

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")

            # If JSON parsing fails or no content found, return raw text
            yield api_response.text

        except httpx.HTTPStatusError as e:
            error_text = f"HTTP error {e.response.status_code}: {e.response.text}"
            print(error_text)

            yield error_text
        except httpx.RequestError as e:
            error_text = f"Request error: {str(e)}"
            print(error_text)
            yield error_text
        except Exception as e:
            error_text = f"Error with request: {str(e)}"
            print(error_text)
            yield error_text

    def execute(self, prompt, stream, response, conversation):
        """
        Execute the GitHub Copilot completion.

        Args:
            prompt: The user prompt
            stream: Whether to stream the response
            response: The response object
            conversation: The conversation history

        Yields:
            Generated text content
        """
        # Get API key
        try:
            api_key = self.authenticator.get_api_key()
        except Exception as e:
            yield f"Error getting GitHub Copilot API key: {str(e)}"
            return

        # Get model name
        model_name = self._get_model_for_api(self.model_id)
        # Prepare the request with required headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "editor-version": "vscode/1.85.1",
            "Copilot-Integration-Id": "vscode-chat",  # Use a recognized integration ID
        }

        # Build conversation messages
        messages = self._build_conversation_messages(prompt, conversation)

        # Get options
        max_tokens = prompt.options.max_tokens or 8192
        temperature = prompt.options.temperature or 0.1

        # Prepare payload
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": model_name in self.get_streaming_models(),
        }

        # Check if model supports streaming
        supports_streaming = model_name in self.get_streaming_models()

        # Check if model supports streaming
        if supports_streaming and stream:
            payload["stream"] = True
            yield from self._handle_streaming_request(
                prompt, headers, payload, model_name
            )
        else:
            # Use non-streaming request for unsupported models or when streaming is disabled
            payload["stream"] = False
            yield from self._non_streaming_request(prompt, headers, payload, model_name)

    def _build_conversation_messages(
        self, prompt, conversation
    ) -> List[Dict[str, str]]:
        """
        Build the messages array from conversation history.

        Args:
            prompt: The current prompt
            conversation: The conversation history

        Returns:
            List of message dictionaries
        """
        messages = []

        # Extract messages from conversation history
        if conversation and conversation.responses:
            for prev_response in conversation.responses:
                # Add user message
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                # Add assistant message
                messages.append({"role": "assistant", "content": prev_response.text()})

        # Add the current prompt and system message if needed
        if messages:
            # Add system message if not present
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": self.DEFAULT_SYSTEM_MESSAGE,
                    },
                )
            # Add the current prompt
            messages.append({"role": "user", "content": prompt.prompt})
        else:
            # First message in conversation
            messages = [
                {
                    "role": "system",
                    "content": self.DEFAULT_SYSTEM_MESSAGE,
                },
                {"role": "user", "content": prompt.prompt},
            ]

        return messages

    def _handle_streaming_request(self, prompt, headers, payload, model_name):
        """
        Handle streaming requests to the API.

        Args:
            prompt: The user prompt
            headers: Request headers
            payload: Request payload
            model_name: The model name for logging

        Yields:
            Generated text content
        """
        try:
            with httpx.Client() as client:
                with client.stream(
                    "POST",
                    f"{self.API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.DEFAULT_TIMEOUT,
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue

                        # Handle both bytes and string types
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")

                        line = line.strip()
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if data == "[DONE]":
                                continue

                            try:
                                json_data = json.loads(data)
                                if "choices" in json_data and json_data["choices"]:
                                    choice = json_data["choices"][0]

                                    # Handle different response formats
                                    if "delta" in choice:
                                        content = choice["delta"].get("content", "")
                                        if content:
                                            yield content
                                    elif "text" in choice:
                                        content = choice.get("text", "")
                                        if content:
                                            yield content
                                    elif "message" in choice:
                                        content = choice["message"].get("content", "")
                                        if content:
                                            yield content
                            except json.JSONDecodeError:
                                # If not valid JSON, check if it's plain text content
                                if (
                                    data
                                    and not data.startswith("{")
                                    and not data.startswith("[")
                                ):
                                    yield data

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            print(error_msg)
            # Print more detailed error information
            print(f"Request headers: {headers}")
            print(f"Request payload: {json.dumps(payload)}")
            # Fall back to non-streaming on error
            payload["stream"] = False
            yield from self._non_streaming_request(prompt, headers, payload, model_name)
        except httpx.RequestError as e:
            print(f"Request error: {str(e)}")
            # Fall back to non-streaming on error
            payload["stream"] = False
            yield from self._non_streaming_request(prompt, headers, payload, model_name)
        except Exception as e:
            print(f"Error with streaming request: {str(e)}")
            # Fall back to non-streaming on error
            payload["stream"] = False
            yield from self._non_streaming_request(prompt, headers, payload, model_name)
