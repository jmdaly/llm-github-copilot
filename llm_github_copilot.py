import llm
import os
import json
import time
import httpx
from datetime import datetime
from typing import Optional, Dict, Any, List, Generator
from pydantic import Field, field_validator


@llm.hookimpl
def register_models(register):
    # Register the main model
    register(GitHubCopilot())
    
    # Register all model variants
    # Claude variants
    claude_3_5_sonnet = GitHubCopilot()
    claude_3_5_sonnet.model_id = "github-copilot/claude-3-5-sonnet"
    register(claude_3_5_sonnet)
    
    claude_3_7_sonnet = GitHubCopilot()
    claude_3_7_sonnet.model_id = "github-copilot/claude-3-7-sonnet"
    register(claude_3_7_sonnet)
    
    claude_3_7_sonnet_thought = GitHubCopilot()
    claude_3_7_sonnet_thought.model_id = "github-copilot/claude-3-7-sonnet-thought" 
    register(claude_3_7_sonnet_thought)
    
    # OpenAI models
    o1_model = GitHubCopilot()
    o1_model.model_id = "github-copilot/o1"
    register(o1_model)
    
    o3_mini = GitHubCopilot()
    o3_mini.model_id = "github-copilot/o3-mini" 
    register(o3_mini)
    
    # Google models
    gemini = GitHubCopilot()
    gemini.model_id = "github-copilot/gemini-2.0-flash-001"
    register(gemini)


class GitHubCopilotAuthenticator:
    """
    Handles authentication with GitHub Copilot using device code flow.
    """
    def __init__(self) -> None:
        # Constants for GitHub API
        self.github_client_id = "Iv1.b507a08c87ecfe98"  # GitHub Copilot client ID
        self.github_device_code_url = "https://github.com/login/device/code"
        self.github_access_token_url = "https://github.com/login/oauth/access_token"
        self.github_api_key_url = "https://api.github.com/copilot_internal/v2/token"
        
        # Token storage paths
        self.token_dir = os.getenv(
            "GITHUB_COPILOT_TOKEN_DIR",
            os.path.expanduser("~/.config/llm/github_copilot")
        )
        self.access_token_file = os.path.join(
            self.token_dir,
            os.getenv("GITHUB_COPILOT_ACCESS_TOKEN_FILE", "access-token")
        )
        self.api_key_file = os.path.join(
            self.token_dir, 
            os.getenv("GITHUB_COPILOT_API_KEY_FILE", "api-key.json")
        )
        self._ensure_token_dir()

    def _ensure_token_dir(self) -> None:
        """Ensure the token directory exists."""
        if not os.path.exists(self.token_dir):
            os.makedirs(self.token_dir, exist_ok=True)

    def _get_github_headers(self, access_token: Optional[str] = None) -> Dict[str, str]:
        """Generate standard GitHub headers for API requests."""
        headers = {
            "accept": "application/json",
            "editor-version": "vscode/1.85.1",
            "editor-plugin-version": "copilot/1.155.0",
            "user-agent": "GithubCopilot/1.155.0",
            "accept-encoding": "gzip,deflate,br",
        }
        
        if access_token:
            headers["authorization"] = f"token {access_token}"
            
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
            
        return headers

    def get_access_token(self) -> str:
        """
        Get GitHub access token, refreshing if necessary.
        """
        try:
            with open(self.access_token_file, "r") as f:
                access_token = f.read().strip()
                if access_token:
                    return access_token
        except IOError:
            pass

        # No valid token found, need to login
        for attempt in range(3):
            try:
                access_token = self._login()
                try:
                    with open(self.access_token_file, "w") as f:
                        f.write(access_token)
                except IOError:
                    print("Error saving access token to file")
                return access_token
            except Exception as e:
                print(f"Login attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # Last attempt
                    raise Exception("Failed to get access token after 3 attempts")
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
                else:
                    print("API key expired, refreshing")
        except (IOError, json.JSONDecodeError, KeyError):
            pass

        try:
            api_key_info = self._refresh_api_key()
            with open(self.api_key_file, "w") as f:
                json.dump(api_key_info, f)
            return api_key_info.get("token")
        except Exception as e:
            raise Exception(f"Failed to get API key: {str(e)}")

    def _get_device_code(self) -> Dict[str, str]:
        """
        Get a device code for GitHub authentication.
        """
        try:
            client = httpx.Client()
            resp = client.post(
                self.github_device_code_url,
                headers=self._get_github_headers(),
                json={"client_id": self.github_client_id, "scope": "read:user"},
            )
            resp.raise_for_status()
            resp_json = resp.json()

            required_fields = ["device_code", "user_code", "verification_uri"]
            if not all(field in resp_json for field in required_fields):
                raise Exception("Response missing required fields")
                
            return resp_json
        except Exception as e:
            raise Exception(f"Failed to get device code: {str(e)}")

    def _poll_for_access_token(self, device_code: str) -> str:
        """
        Poll for an access token after user authentication.
        """
        client = httpx.Client()
        max_attempts = 12  # 1 minute (12 * 5 seconds)
        
        for attempt in range(max_attempts):
            try:
                resp = client.post(
                    self.github_access_token_url,
                    headers=self._get_github_headers(),
                    json={
                        "client_id": self.github_client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                resp.raise_for_status()
                resp_json = resp.json()

                if "access_token" in resp_json:
                    print("Authentication successful!")
                    return resp_json["access_token"]
                elif "error" in resp_json and resp_json.get("error") == "authorization_pending":
                    print(f"Waiting for authorization... (attempt {attempt+1}/{max_attempts})")
                else:
                    print(f"Unexpected response: {resp_json}")
            except Exception as e:
                raise Exception(f"Failed to get access token: {str(e)}")
                
            time.sleep(5)
            
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
                    self.github_api_key_url, 
                    headers=headers
                )
                response.raise_for_status()

                response_json = response.json()

                if "token" in response_json:
                    return response_json
                else:
                    print(f"API key response missing token: {response_json}")
            except Exception as e:
                print(f"Error refreshing API key (attempt {attempt+1}/{max_retries}): {str(e)}")

            if attempt < max_retries - 1:
                time.sleep(1)

        raise Exception("Failed to refresh API key after maximum retries")


class GitHubCopilot(llm.Model):
    """
    GitHub Copilot model implementation for LLM.
    """
    model_id = "github-copilot"
    can_stream = True
    
    # Map of model names to API model identifiers
    MODEL_MAPPINGS = {
        "github-copilot": "gpt-4o",
        "github-copilot/o1": "o1",
        "github-copilot/o3-mini": "o3-mini",
        "github-copilot/gemini-2.0-flash-001": "gemini-2.0-flash-001",
        "github-copilot/claude-3-5-sonnet": "claude-3.5-sonnet",  # Note: Fixed to use correct format with dot
        "github-copilot/claude-3-7-sonnet": "claude-3.7-sonnet",  # Note: Fixed to use correct format with dot
        "github-copilot/claude-3-7-sonnet-thought": "claude-3.7-sonnet-thought",  # Note: Fixed to use correct format with dot
    }
    
    # Identify models that need special handling for streaming
    CLAUDE_MODELS = [
        "claude-3.5-sonnet",
        "claude-3.7-sonnet", 
        "claude-3.7-sonnet-thought",
    ]
    
    class Options(llm.Options):
        """
        Options for the GitHub Copilot model.
        """
        max_tokens: Optional[int] = Field(
            description="Maximum number of tokens to generate",
            default=1024
        )
        temperature: Optional[float] = Field(
            description="Controls randomness in the output",
            default=0.7
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
        self.authenticator = GitHubCopilotAuthenticator()
        # GitHub Copilot API base URL
        self.api_base = "https://api.githubcopilot.com"
        
    def _get_model_for_api(self, model: str) -> str:
        """Convert model name to API-compatible format."""
        # Strip provider prefix if present
        if '/' in model:
            _, model_name = model.split('/', 1)
            if model_name in self.MODEL_MAPPINGS.values():
                return model_name
        
        # Use the mapping or default to gpt-4o
        return self.MODEL_MAPPINGS.get(model, "gpt-4o")
    
    def _is_claude_model(self, model_name: str) -> bool:
        """Check if model is a Claude model that needs special handling."""
        return model_name in self.CLAUDE_MODELS
        
    def execute(self, prompt, stream, response, conversation):
        """
        Execute the GitHub Copilot completion.
        """
        # Get API key
        try:
            api_key = self.authenticator.get_api_key()
        except Exception as e:
            yield f"Error getting GitHub Copilot API key: {str(e)}"
            return
        
        # Get model name
        model_name = self._get_model_for_api(self.model_id)
        # For debugging
        print(f"Using model ID: {self.model_id}, API model name: {model_name}")
        
        # Prepare the request with required headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "editor-version": "vscode/1.85.1",
            "editor-plugin-version": "copilot/1.155.0",
            "user-agent": "GithubCopilot/1.155.0",
            "Copilot-Integration-Id": "vscode-chat",  # Use a recognized integration ID
        }
        
        # Extract messages from conversation
        messages = []
        if conversation and conversation.responses:
            for prev_response in conversation.responses:
                # Add user message
                messages.append({
                    "role": "user",
                    "content": prev_response.prompt.prompt
                })
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": prev_response.text()
                })
                
        # Add the current prompt
        if messages:
            # Add system message if not present
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": "You are GitHub Copilot, an AI programming assistant."
                })
            # Add the current prompt
            messages.append({
                "role": "user",
                "content": prompt.prompt
            })
        else:
            # First message in conversation
            messages = [
                {
                    "role": "system",
                    "content": "You are GitHub Copilot, an AI programming assistant."
                },
                {
                    "role": "user",
                    "content": prompt.prompt
                }
            ]
            
        # Get options
        max_tokens = prompt.options.max_tokens or 1024
        temperature = prompt.options.temperature or 0.7
        
        # Prepare payload - based on litellm, we'll use a simpler approach
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        
        # Record additional information in response_json
        response.response_json = {
            "model": model_name,
            "messages": messages,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        client = httpx.Client()
        
        # Try non-streaming first for all models
        if not stream:
            try:
                print(f"Sending non-streaming request to {self.api_base}/chat/completions with model {model_name}")
                api_response = client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                api_response.raise_for_status()
                
                print(f"Response status: {api_response.status_code}")
                print(f"Response content type: {api_response.headers.get('content-type', 'none')}")
                
                try:
                    # Parse the JSON response
                    json_data = api_response.json()
                    print(f"Response JSON keys: {list(json_data.keys())}")
                    
                    # Extract the message content
                    if "choices" in json_data and json_data["choices"]:
                        choice = json_data["choices"][0]
                        print(f"Choice keys: {list(choice.keys())}")
                        
                        content = None
                        if "message" in choice:
                            message = choice["message"]
                            print(f"Message keys: {list(message.keys())}")
                            content = message.get("content", "")
                        
                        # Update usage statistics in response_json if available
                        if "usage" in json_data:
                            response.response_json["usage"] = json_data["usage"]
                        
                        if content:
                            yield content
                        else:
                            print("No content found in structured response")
                            yield "No content found in the response"
                    else:
                        print("No choices found in response")
                        yield "No content found in the response"
                        
                except json.JSONDecodeError:
                    # If response is not JSON, return the raw text
                    print("Response is not valid JSON")
                    raw_text = api_response.text
                    print(f"Raw text (first 100 chars): {raw_text[:100]}")
                    yield raw_text
                    
            except httpx.HTTPStatusError as e:
                print(f"HTTP error: {e}")
                error_detail = e.response.text if hasattr(e, 'response') and e.response else "No response details"
                yield f"HTTP error: {str(e)} - {error_detail}"
                return
                
            except Exception as e:
                print(f"Exception during request: {e}")
                yield f"Error with GitHub Copilot request: {str(e)}"
                return
        else:
            # Handle streaming
            try:
                print(f"Sending streaming request to {self.api_base}/chat/completions with model {model_name}")
                with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                ) as stream_response:
                    print(f"Stream response status: {stream_response.status_code}")
                    stream_response.raise_for_status()
                    
                    # For debugging purposes, collect some of the initial response
                    debug_chunks = []
                    
                    try:
                        # Read the raw content to debug
                        for i, chunk in enumerate(stream_response.iter_raw()):
                            if i < 5:  # Just collect the first few chunks for debugging
                                debug_chunks.append(chunk)
                                print(f"Raw chunk {i}: {chunk[:50]}")
                            
                            try:
                                # Try to decode as text
                                chunk_str = chunk.decode('utf-8')
                                
                                # Check for SSE format
                                for line in chunk_str.splitlines():
                                    if line.startswith("data: "):
                                        data = line[6:]
                                        if data == "[DONE]":
                                            continue
                                        
                                        try:
                                            json_data = json.loads(data)
                                            if "choices" in json_data and json_data["choices"]:
                                                delta = json_data["choices"][0].get("delta", {})
                                                content = delta.get("content", "")
                                                if content:
                                                    yield content
                                        except json.JSONDecodeError:
                                            if data and data != "[DONE]":
                                                yield data
                                    elif line.strip():  # If not SSE but has content
                                        yield line
                            except UnicodeDecodeError:
                                # Not utf-8 decodable, skip
                                pass
                    except Exception as chunk_error:
                        print(f"Error processing stream chunks: {chunk_error}")
                        
                        # If we have debug chunks, log them
                        if debug_chunks:
                            print(f"Debug chunks collected: {len(debug_chunks)}")
                            for i, chunk in enumerate(debug_chunks):
                                print(f"Debug chunk {i}: {chunk}")
                        
                        # Try one more time with a fallback approach
                        try:
                            # Reset the stream
                            print("Trying fallback approach for streaming...")
                            
                            # Don't use streaming for the fallback
                            fallback_payload = dict(payload)
                            fallback_payload["stream"] = False
                            
                            fallback_response = client.post(
                                f"{self.api_base}/chat/completions",
                                headers=headers,
                                json=fallback_payload,
                                timeout=120
                            )
                            
                            if fallback_response.status_code == 200:
                                try:
                                    json_data = fallback_response.json()
                                    if "choices" in json_data and json_data["choices"]:
                                        message = json_data["choices"][0].get("message", {})
                                        content = message.get("content", "")
                                        
                                        if content:
                                            yield content
                                        else:
                                            yield "No content found in fallback response"
                                except:
                                    yield fallback_response.text
                            else:
                                yield f"Fallback request failed with status {fallback_response.status_code}"
                        except Exception as fallback_error:
                            yield f"Fallback also failed: {str(fallback_error)}"
                            
            except httpx.HTTPStatusError as e:
                print(f"HTTP error during streaming: {e}")
                error_detail = e.response.text if hasattr(e, 'response') and e.response else "No response details"
                yield f"HTTP error during streaming: {str(e)} - {error_detail}"
                return
                
            except Exception as e:
                print(f"Exception during streaming: {e}")
                yield f"Error with GitHub Copilot streaming request: {str(e)}"
                return

    def _execute_claude_model(self, prompt, stream, response, headers, payload):
        """
        Special handling for Claude model completion requests.
        """
        client = httpx.Client()
        
        # For Claude models, always use non-streaming requests for now
        try:
            print("Sending request to Claude model...")
            api_response = client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180  # Claude may take longer
            )
            
            # Print debugging information
            print(f"Claude response status: {api_response.status_code}")
            print(f"Claude response content-type: {api_response.headers.get('content-type', 'none')}")
            
            # Check if the request was successful
            if api_response.status_code != 200:
                print(f"Error response from Claude API: {api_response.text}")
                yield f"Error from Claude model: HTTP {api_response.status_code} - {api_response.text}"
                return
                
            # Debug: Log the raw response
            raw_response = api_response.text
            print(f"Raw Claude response (first 200 chars): {raw_response[:200]}...")
            
            # Parse the JSON response
            try:
                json_data = api_response.json()
                print(f"Claude response JSON keys: {json_data.keys()}")
                
                # Extract the message content
                if "choices" in json_data and json_data["choices"]:
                    print(f"Found choices in response: {len(json_data['choices'])}")
                    choice = json_data["choices"][0]
                    print(f"First choice keys: {choice.keys()}")
                    
                    # Check for message or content directly
                    content = None
                    if "message" in choice:
                        message = choice["message"]
                        print(f"Message keys: {message.keys()}")
                        content = message.get("content", "")
                    elif "content" in choice:
                        content = choice.get("content", "")
                    else:
                        # Try other potential keys where content might be found
                        for key in choice.keys():
                            if isinstance(choice[key], dict) and "content" in choice[key]:
                                content = choice[key]["content"]
                                break
                            elif isinstance(choice[key], str) and len(choice[key]) > 10:
                                content = choice[key]
                                break
                    
                    # If no content found through standard paths, check the entire response
                    if not content:
                        print("Searching for content in response...")
                        # Try to find anything that looks like content
                        if isinstance(json_data, dict):
                            for key, value in json_data.items():
                                if key == "content" and isinstance(value, str):
                                    content = value
                                    break
                                elif isinstance(value, dict) and "content" in value and isinstance(value["content"], str):
                                    content = value["content"]
                                    break
                    
                    # Update usage statistics in response_json if available
                    if "usage" in json_data:
                        response.response_json["usage"] = json_data["usage"]
                    
                    if content:
                        print(f"Found content (length: {len(content)})")
                        # If streaming is requested, simulate streaming by yielding chunks
                        if stream:
                            # Simple chunking by sentences or paragraphs
                            import re
                            chunks = re.split(r'(?<=[.!?])\s+', content)
                            for chunk in chunks:
                                if chunk.strip():
                                    yield chunk + " "
                                    time.sleep(0.02)  # Small delay to simulate streaming
                        else:
                            # Return the full content at once
                            yield content
                    else:
                        print("No content found in structured JSON response")
                        # As a fallback, return the raw response if it looks like text
                        if raw_response and len(raw_response) > 10 and not raw_response.startswith('{'):
                            yield raw_response
                        else:
                            yield "No content found in the Claude model response"
                else:
                    print("No choices found in response")
                    # Try to extract content from elsewhere in the response
                    if "content" in json_data:
                        yield json_data["content"]
                    else:
                        yield f"No structured content found. Raw response: {raw_response[:500]}"
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                # If we can't parse JSON, return the raw response
                if raw_response and len(raw_response) > 10 and not raw_response.startswith('{'):
                    yield raw_response
                else:
                    yield f"Couldn't parse Claude response as JSON. Raw response: {raw_response[:500]}..."
                    
        except Exception as e:
            print(f"Exception handling Claude request: {str(e)}")
            yield f"Error with Claude model request: {str(e)}"
            return
