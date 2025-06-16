import llm
import llm.default_plugins.openai_models
import os
import json
import time
from pathlib import Path
import httpx
from datetime import datetime, timezone
from typing import Optional, Any
import click


def _fetch_models_data(authenticator: "GitHubCopilotAuthenticator") -> dict:
    """
    Helper function to fetch raw model data from the GitHub Copilot API.

    This function makes an HTTP request to the GitHub Copilot API to retrieve
    information about available models. It handles authentication and error reporting.

    Args:
        authenticator: The GitHubCopilotAuthenticator instance to use for authentication

    Returns:
        dict: The raw JSON response from the API containing model data

    Raises:
        Exception: If the API request fails for any reason
    """
    try:
        api_key = authenticator.get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "editor-version": "vscode/1.85.1",
        }
        response = httpx.get(
            "https://api.githubcopilot.com/models", headers=headers, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        # Ensure the response body is read before accessing .text,
        # in case of chunked error responses.
        e.response.read()
        error_body_text = e.response.text
        print(
            f"Error fetching models data (HTTP {e.response.status_code}): {error_body_text}"
        )
        raise
    except httpx.RequestError as e:
        print(f"Error fetching models data (Request Error): {str(e)}")
        raise
    except Exception as e:
        # Catch potential errors from get_api_key() as well
        print(f"Error fetching models data: {str(e)}")
        raise

_models_data_cache = None
def fetch_models_data(authenticator: "GitHubCopilotAuthenticator") -> dict:
    """
    Cached/memoized wrapper around _fetch_models_data
    """
    global _models_data_cache
    if _models_data_cache is None:
        _models_data_cache = _fetch_models_data(authenticator)  # Use helper
    return _models_data_cache


@llm.hookimpl
def register_models(register):
    """Register all GitHub Copilot models with the LLM CLI tool."""
    # Try to fetch available models without forcing authentication
    try:
        # Create an authenticator to fetch models
        authenticator = GitHubCopilotAuthenticator()

        # Only fetch models if we already have valid credentials
        if authenticator.has_valid_credentials():
            models_data = fetch_models_data(authenticator)

            # Register all model variants
            for model_data in models_data.get("data", []):
                model_id = f"github_copilot/{model_data.get('id')}"

                register(
                    GitHubCopilotChat(model_id, model_data),
                    GitHubCopilotAsyncChat(model_id, model_data),
                )
    except Exception as e:
        print(f"Warning: Failed to fetch GitHub Copilot models: {str(e)}")


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

    # Key identifiers for LLM key storage
    ACCESS_TOKEN_KEY = "github_copilot_access_token"
    DEFAULT_KEYS_JSON_CONTENT = {"// Note": "This file stores secret API credentials. Do not share!"}

    def __init__(self) -> None:
        # Token storage paths for API key (still using file for this)
        self.token_dir = Path(llm.user_dir())
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.api_key_file = self.token_dir / os.getenv(
            "GITHUB_COPILOT_API_KEY_FILE", "github_copilot_api_key.json"
        )

    def has_valid_credentials(self) -> bool:
        """
        Check if we have valid API credentials without triggering authentication.

        This method checks for valid credentials in the following order:
        1. Checks if a valid API key file exists with a non-expired token
        2. Checks if a valid access token exists in the LLM key storage

        Returns:
            bool: True if valid credentials exist, False otherwise

        Note:
            This method does not attempt to refresh or obtain new credentials.
        """
        # Order of checks:
        # 1. Valid (non-expired, token exists and is non-empty) API key file
        # 2. Environment variables for access token
        # 3. LLM-stored access token (non-empty)

        # Check 1: API Key File
        try:
            if self.api_key_file.exists():
                api_key_info = json.loads(self.api_key_file.read_text())
                # Ensure api_key_info is a dict, token exists and is not empty, and is not expired
                if isinstance(api_key_info, dict) and \
                   api_key_info.get("token") and \
                   isinstance(api_key_info.get("token"), str) and \
                   api_key_info.get("token").strip() and \
                   api_key_info.get("expires_at", 0) > datetime.now().timestamp():
                    return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError, AttributeError, TypeError):
            # Ignore errors related to API key file processing, proceed to next checks
            pass

        # Check 2: Environment Variables
        for env_var in ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]:
            env_token = os.environ.get(env_var)
            if env_token and env_token.strip():
                return True

        # Check 3: LLM-stored Access Token
        try:
            access_token = llm.get_key(None, self.ACCESS_TOKEN_KEY)
            if access_token and isinstance(access_token, str) and access_token.strip():
                return True
        except Exception: # Catching broad exception from llm.get_key if it fails
            pass

        return False

    def _get_github_headers(self, access_token: Optional[str] = None) -> dict[str, str]:
        """
        Generate standard GitHub headers for API requests.

        Creates a dictionary of HTTP headers required for GitHub API requests,
        optionally including an authorization header with the provided access token.

        Args:
            access_token: Optional GitHub access token to include in the headers

        Returns:
            dict[str, str]: Dictionary of HTTP headers for GitHub API requests
        """
        headers = self.DEFAULT_HEADERS.copy()

        if access_token:
            headers["authorization"] = f"token {access_token}"

        return headers

    def get_access_token(self) -> str:
        """
        Get GitHub access token, refreshing if necessary.

        First checks for GH_COPILOT_KEY environment variable,
        then falls back to the LLM key storage.
        """
        # Check environment variables first
        for env_var in ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]:
            env_token = os.environ.get(env_var)
            if env_token and env_token.strip():
                return env_token.strip()

        # Try to read existing token from LLM key storage
        try:
            access_token = llm.get_key(None, self.ACCESS_TOKEN_KEY)
            if access_token:
                return access_token
        except Exception:
            pass

        # No valid token found, inform user they need to authenticate
        raise Exception(
            "GitHub Copilot authentication required. Run 'llm github_copilot auth login' to authenticate or set the GH_COPILOT_KEY environment variable."
        )

    def get_api_key(self) -> str:
        """
        Get the API key, refreshing if necessary.
        """
        try:
            api_key_info = json.loads(self.api_key_file.read_text())
            if api_key_info.get("expires_at", 0) > datetime.now().timestamp():
                return api_key_info.get("token")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        # If we don't have a valid API key, check if we need to authenticate first
        try:
            access_token = llm.get_key("github_copilot", self.ACCESS_TOKEN_KEY)
        except (TypeError, Exception):
            access_token = None

        if not access_token and not (
            os.environ.get("GH_COPILOT_TOKEN") or os.environ.get("GITHUB_COPILOT_TOKEN")
        ):
            raise Exception(
                "GitHub Copilot authentication required. Run 'llm github_copilot auth login' first."
            )

        try:
            api_key_info = self._refresh_api_key()
            self.api_key_file.write_text(json.dumps(api_key_info), encoding="utf-8")
            self.api_key_file.chmod(0o600)
            return api_key_info.get("token")
        except Exception as e:
            raise Exception(f"Failed to get API key: {str(e)}")

    def _get_device_code(self) -> dict[str, str]:
        """
        Get a device code for GitHub authentication using the device flow.

        This method initiates the GitHub device flow authentication process by
        requesting a device code from the GitHub API. The device code is used
        to associate the user's browser session with this application.

        Returns:
            dict[str, str]: A dictionary containing the device code, user code,
                           verification URI, and other information needed for
                           the authentication flow

        Raises:
            Exception: If the request fails or the response is missing required fields
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

        This method repeatedly polls the GitHub API to check if the user has
        completed the authentication process in their browser. It continues
        polling until either the user completes authentication, an error occurs,
        or the maximum number of polling attempts is reached.

        Args:
            device_code: The device code obtained from _get_device_code()

        Returns:
            str: The GitHub access token if authentication is successful

        Raises:
            Exception: If polling times out or an error occurs during polling
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

        This method orchestrates the complete GitHub device flow authentication process:
        1. Obtains a device code and user code
        2. Displays instructions for the user to complete authentication in their browser
        3. Polls for the access token until the user completes authentication

        Returns:
            str: The GitHub access token if authentication is successful

        Raises:
            Exception: If any step of the authentication process fails
        """
        device_code_info = self._get_device_code()

        device_code = device_code_info["device_code"]
        user_code = device_code_info["user_code"]
        verification_uri = device_code_info["verification_uri"]

        click.echo("Fetching API key...")
        print(
            f"\nPlease visit {verification_uri} and enter code {user_code} to authenticate GitHub Copilot.\n"
        )

        return self._poll_for_access_token(device_code)

    def _refresh_api_key(self, access_token_override: Optional[str] = None) -> dict[str, Any]:
        """
        Refresh the API key using the access token.

        Args:
            access_token_override: If provided, use this access token instead of
                                   calling get_access_token().

        This method exchanges a GitHub access token for a GitHub Copilot API key
        by making a request to the GitHub Copilot API. It includes retry logic
        to handle transient failures.

        Returns:
            dict[str, Any]: A dictionary containing the API key and its expiration time
                           in the format {"token": "api_key", "expires_at": timestamp}

        Raises:
            Exception: If the API key cannot be refreshed after maximum retries
        """
        access_token = access_token_override or self.get_access_token()
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

    def _get_github_user_info(self, access_token_override: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Fetch user information from GitHub API using the access token.

        Args:
            access_token_override: If provided, use this access token instead of
                                   calling get_access_token().

        Returns:
            dict[str, Any]: Dictionary containing user information (e.g., login)
                            or None if fetching fails.
        """
        try:
            access_token = access_token_override or self.get_access_token()
            headers = self._get_github_headers(access_token)
            client = httpx.Client()
            response = client.get(
                "https://api.github.com/user", headers=headers, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Failed to fetch GitHub user info: {str(e)}")
            return None


class _GitHubCopilot():
    """
    GitHub Copilot model implementation for LLM.
    """
    API_BASE = "https://api.githubcopilot.com"

    def __init__(self, model_id, model_data, *args, **kwargs):
        # Get API key
        self.authenticator = GitHubCopilotAuthenticator()
        try:
            api_key = self.authenticator.get_api_key()
        except Exception as e:
            error_message = str(e)
            if "authentication required" in error_message.lower():
                print("GitHub Copilot authentication required. Run 'llm github_copilot auth login' to authenticate.")
            else:
                print(f"Error getting GitHub Copilot API key: {error_message}")
            raise
        headers = self.authenticator._get_github_headers()
        if "/" in model_id:
            _, model_name = model_id.split("/", 1)
        else:
            model_name = model_id

        supports = model_data.get("capabilities", {}).get("supports", {})

        super().__init__(
                model_id,
                *args,
                key=api_key,
                model_name=model_name,
                api_base=self.API_BASE,
                headers=headers,
                can_stream=supports.get("streaming", False),
                supports_tools=supports.get("tool_calls", False),
                supports_schema=supports.get("structured_outputs", False),
                **kwargs)

    def __str__(self):
        return "GitHub Copilot Chat: {}".format(self.model_id)

class GitHubCopilotChat(_GitHubCopilot, llm.default_plugins.openai_models.Chat):
    pass

class GitHubCopilotAsyncChat(_GitHubCopilot, llm.default_plugins.openai_models.AsyncChat):
    pass


# LLM CLI command implementation
@llm.hookimpl
def register_commands(cli):
    @cli.group(name="github_copilot")
    def github_copilot_group():
        """
        Commands relating to the llm-github_copilot plugin

        By default github_copilot will use the following in precedence order to obtain an access_token for communication :
            $GH_COPILOT_KEY
            llm keystore for github_copilot
        """
        pass

    @github_copilot_group.group(name="auth")
    def auth_group():
        """
        Manage GitHub Copilot authentication.

        """
        pass

    @auth_group.command(name="login")
    @click.option(
        "-f", "--force", is_flag=True, help="Force login even if already authenticated"
    )
    @click.option(
        "--show-only", is_flag=True, help="Perform login but only display tokens, do not save them"
    )
    def login_command(force, show_only):
        """
        Authenticate with GitHub Copilot to generate a new access token.
        """
        authenticator = GitHubCopilotAuthenticator()

        # Check if already authenticated and not forcing re-login
        if not force and authenticator.has_valid_credentials():
            click.echo("Valid GitHub Copilot authentication already exists.")
            click.echo("Use --force to re-authenticate if needed.")
            click.echo(
                "Run 'llm github_copilot auth status' to see current authentication details."
            )
            return 0

        try:
            # Check if using environment variables
            env_var_used = None
            for env_var in ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]:
                if os.environ.get(env_var):
                    env_var_used = env_var
                    break

            if env_var_used:
                # Using environment variable for access token
                access_token = os.environ.get(env_var_used).strip()
                click.echo(
                    f"Not possible to initiate login with environment variable {env_var_used} set, please unset"
                )
                click.echo(
                    f"Access token: {access_token} (from environment variable {env_var_used})"
                )
                return 0
            else:
                # Start the login process
                click.echo(
                    "Starting GitHub Copilot login process to obtain an access_token & API key..."
                )
                access_token = authenticator._login() # This performs the device flow

                # Get the API key using the newly obtained access_token
                api_key_info = authenticator._refresh_api_key(access_token_override=access_token)

                if show_only:
                    user_info = authenticator._get_github_user_info(access_token_override=access_token)
                    user_login = "<unable to fetch>"
                    if user_info and "login" in user_info:
                        user_login = user_info['login']
                    
                    api_key_value = api_key_info.get("token", "<not available>")
                    expires_at = api_key_info.get("expires_at", 0)
                    expiry_date_str = "never"
                    if expires_at > 0:
                        expiry_date_str = datetime.fromtimestamp(expires_at, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                    click.echo("GitHub Copilot: ✓ Authenticated")
                    click.echo(f"          User: {user_login}")
                    click.echo(f"   AccessToken: Valid")
                    click.echo(f"   AccessToken: {access_token}")
                    click.echo(f"       API Key: Valid, expires {expiry_date_str}")
                    click.echo(f"       API key: {api_key_value}")
                    click.echo("")
                    click.echo("You can set GH_COPILOT_TOKEN or GITHUB_COPILOT_TOKEN to the token value above.")
                    click.echo("Note: These tokens have NOT been saved to the LLM keystore or API key file.")

                else: # Save the tokens
                    # Save the access token to LLM key storage
                    try:
                        keys_path = llm.user_dir() / "keys.json"
                        key_name_to_set = authenticator.ACCESS_TOKEN_KEY
                        
                        current_keys = {}
                        if keys_path.exists():
                            try:
                                current_keys = json.loads(keys_path.read_text())
                                if not isinstance(current_keys, dict):
                                    current_keys = authenticator.DEFAULT_KEYS_JSON_CONTENT.copy()
                            except json.JSONDecodeError:
                                click.echo(f"Warning: {keys_path} is not valid JSON. Initializing a new one.", err=True)
                                current_keys = authenticator.DEFAULT_KEYS_JSON_CONTENT.copy()
                        else:
                            current_keys = authenticator.DEFAULT_KEYS_JSON_CONTENT.copy()

                        current_keys[key_name_to_set] = access_token
                        
                        keys_path.parent.mkdir(parents=True, exist_ok=True)
                        keys_path.write_text(json.dumps(current_keys, indent=2) + "\n")
                        # Ensure correct permissions if file was newly created by write_text
                        # (chmod might not be needed if umask is set correctly, but explicit is safer)
                        if not os.access(keys_path, os.R_OK | os.W_OK): # Basic check, could be more robust
                             try:
                                 keys_path.chmod(0o600)
                             except OSError as e:
                                 click.echo(f"Warning: Could not set permissions on {keys_path}: {e}", err=True)
                        
                        click.echo(f"Access token saved: {access_token}")
                    except Exception as e:
                        click.echo(f"Error saving access token to LLM key storage: {str(e)}")

                    # Save the API key
                    authenticator.api_key_file.write_text(
                        json.dumps(api_key_info), encoding="utf-8"
                    )
                    try:
                        authenticator.api_key_file.chmod(0o600)
                    except OSError as e:
                        click.echo(f"Warning: Could not set permissions on {authenticator.api_key_file}: {e}", err=True)


                    click.echo("GitHub Copilot login process completed successfully!")
                    click.echo("")

                    # Fetch available models
                    click.echo("Fetching available models...")
                    models_data = fetch_models_data(authenticator)
                    model_ids = [m.get("id") for m in models_data.get("data")]
                    click.echo(f"Available models: {', '.join(model_ids)}")

        except Exception as e:
            click.echo(f"Error during authentication: {str(e)}", err=True)
            return 1

        return 0

    @auth_group.command(name="status")
    @click.option("-v", "--verbose", is_flag=True, help="Show verbose details")
    def status_command(verbose):
        """
        Check GitHub Copilot authentication status.
        """
        authenticator = GitHubCopilotAuthenticator()
        is_authenticated = authenticator.has_valid_credentials()

        if is_authenticated:
            click.echo("GitHub Copilot: ✓ Authenticated")

            # API Key Status
            api_key_status_message = "       API Key: "
            api_key_value_for_verbose = None
            try:
                api_key_info = json.loads(authenticator.api_key_file.read_text())
                api_key_value_for_verbose = api_key_info.get("token", "")
                expires_at = api_key_info.get("expires_at", 0)
                if api_key_value_for_verbose and expires_at > datetime.now(timezone.utc).timestamp():
                    expiry_date = datetime.fromtimestamp(expires_at, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    api_key_status_message += f"Valid, expires {expiry_date}"
                else:
                    api_key_status_message += "Expired, will refresh on next request"
            except (FileNotFoundError, json.JSONDecodeError, KeyError, AttributeError):
                api_key_status_message += "Not found or invalid"
            click.echo(api_key_status_message)

            # Access Token Status
            access_token_status_message = "   AccessToken: "
            access_token_value_for_verbose = None
            env_var_used = None # Define env_var_used here to be available for verbose block
            for env_var_ghc in ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]: # Renamed to avoid conflict
                token = os.environ.get(env_var_ghc)
                if token and token.strip():
                    env_var_used = env_var_ghc
                    access_token_value_for_verbose = token.strip()
                    break
            
            _access_token_status_part = "Not found in keystore" # Default
            if env_var_used:
                _access_token_status_part = f"Valid, via env {env_var_used}"
            else:
                try:
                    stored_token = llm.get_key(None, authenticator.ACCESS_TOKEN_KEY)
                    if stored_token:
                        access_token_value_for_verbose = stored_token # Ensure this is set if from keystore
                        _access_token_status_part = f"Valid, via keystore {authenticator.ACCESS_TOKEN_KEY}"
                    # else: it remains "Not found in keystore"
                except Exception:
                    _access_token_status_part = "Error checking keystore"
            access_token_status_message += _access_token_status_part


            if verbose:
                # GitHub User Info
                user_info = authenticator._get_github_user_info()
                user_login = "<unable to fetch>"
                if user_info and "login" in user_info:
                    user_login = user_info['login']
                click.echo(f"          User: {user_login}")

                # AccessToken Status and Value
                # access_token_status_message is like "   AccessToken: Valid, via keystore ..."
                # We want the part after "   AccessToken: "
                parsed_access_token_status_part = access_token_status_message.split("   AccessToken: ", 1)[1]
                click.echo(f"   AccessToken: {parsed_access_token_status_part}")
                click.echo(f"   AccessToken: {access_token_value_for_verbose or 'Not available'}")
                
                # API Key Status and Value
                # api_key_status_message is like "       API Key: Valid, expires ..."
                # We want the part after "       API Key: "
                parsed_api_key_status_part = api_key_status_message.split("       API Key: ", 1)[1]
                click.echo(f"       API Key: {parsed_api_key_status_part}")
                click.echo(f"       API key: {api_key_value_for_verbose or 'Not available'}")
            else: # Not verbose, print the pre-formatted status messages
                click.echo(access_token_status_message)

        else: # This 'else' is for 'if is_authenticated:'
            click.echo("GitHub Copilot: ✗ Not authenticated")
            click.echo(
                "Run 'llm github_copilot auth login' to authenticate or set $GH_COPILOT_TOKEN or $GITHUB_COPILOT_TOKEN."
            )

        return 0

    @auth_group.command(name="refresh")
    @click.option("-v", "--verbose", is_flag=True, help="Show verbose details")
    def refresh_command(verbose):
        """
        Force refresh the GitHub Copilot API key.
        """
        authenticator = GitHubCopilotAuthenticator()

        try:
            # Check if we have an access token
            try:
                access_token = llm.get_key(
                    None, authenticator.ACCESS_TOKEN_KEY
                )
            except Exception:
                access_token = None

            if not access_token and not (
                os.environ.get("GH_COPILOT_TOKEN")
                or os.environ.get("GITHUB_COPILOT_TOKEN")
            ):
                click.echo(
                    "No access token found. Run 'llm github_copilot auth login' first."
                )
                return 1

            # Force refresh the API key
            click.echo("Refreshing API key...")
            api_key_info = authenticator._refresh_api_key()
            authenticator.api_key_file.write_text(
                json.dumps(api_key_info), encoding="utf-8"
            )
            authenticator.api_key_file.chmod(0o600)

            # Show API key information in the same format as status command
            expires_at = api_key_info.get("expires_at", 0)
            if expires_at > 0:
                expiry_date = datetime.fromtimestamp(expires_at, timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                click.echo(f"API key expires: {expiry_date}")
                api_key = api_key_info.get("token", "")

                # Only show the API key in verbose mode
                if verbose:
                    click.echo(f"API key: {api_key}")
            else:
                click.echo(
                    "API key refreshed successfully, but no expiry information found."
                )

            return 0
        except Exception as e:
            click.echo(f"Error refreshing API key: {str(e)}", err=True)
            return 1

    @auth_group.command(name="logout")
    def logout_command():
        """
        Remove GitHub Copilot authentication credentials.
        """
        authenticator = GitHubCopilotAuthenticator()

        # Remove access token from LLM key storage
        try:
            keys_path = llm.user_dir() / "keys.json"
            key_name_to_remove = authenticator.ACCESS_TOKEN_KEY
            
            key_existed_and_removed = False
            if keys_path.exists():
                current_keys = {}
                try:
                    current_keys = json.loads(keys_path.read_text())
                    if not isinstance(current_keys, dict):
                        # Handle case where keys.json is not a dict (e.g. corrupted)
                        # or if it's empty and loads as None or other non-dict type
                        current_keys = {}
                except json.JSONDecodeError:
                    # If file is malformed, treat as if key wasn't there or handle error
                    click.echo(f"Warning: {keys_path} is not valid JSON.", err=True)
                    current_keys = {} # Or re-raise, or return, depending on desired strictness

                if key_name_to_remove in current_keys:
                    del current_keys[key_name_to_remove]
                    # Ensure the // Note comment is preserved if it exists and keys become empty
                    if not current_keys and "// Note" in getattr(GitHubCopilotAuthenticator, "DEFAULT_KEYS_JSON_CONTENT", {}):
                         current_keys["// Note"] = GitHubCopilotAuthenticator.DEFAULT_KEYS_JSON_CONTENT["// Note"]

                    keys_path.write_text(json.dumps(current_keys, indent=2) + "\n")
                    key_existed_and_removed = True
            
            if key_existed_and_removed:
                click.echo("Access token removed from LLM key storage.")
            # If you want to inform the user that the key was not found, uncomment below:
            # else:
            #     click.echo("Access token not found in LLM key storage.")

        except Exception as e:
            click.echo(f"Error removing access token: {str(e)}")

        # Remove API key
        if authenticator.api_key_file.exists():
            authenticator.api_key_file.unlink()
            click.echo("API key removed.")

        click.echo("GitHub Copilot logout completed successfully.")
        return 0

    @github_copilot_group.command(name="models")
    @click.option(
        "-v", "--verbose", is_flag=True, help="Show detailed model information"
    )
    @click.option("--raw", is_flag=True, help="Show raw model details from API (JSON)")
    def models_command(verbose, raw):
        """
        List registered GitHub Copilot models.
        Use -v for details, --raw for full JSON output from API.
        """
        if verbose and raw:
            click.echo(
                "Error: Cannot use both -v and --raw flags simultaneously.", err=True
            )
            # Explicitly return non-zero exit code for error
            ctx = click.get_current_context()
            ctx.exit(1)

        try:
            registered_llm_models = llm.get_models()
            github_model_ids = sorted(
                [
                    model.model_id
                    for model in registered_llm_models
                    if model.model_id.startswith("github_copilot")
                ]
            )

            if not github_model_ids:
                click.echo("No GitHub Copilot models are currently registered.")
                return 0

            # Default output (no flags)
            if not verbose and not raw:
                for model_id in github_model_ids:
                    click.echo(model_id)  # Removed leading "- "
                return  # Implicit exit 0

            # Fetch data if verbose or raw
            authenticator = GitHubCopilotAuthenticator()
            if not authenticator.has_valid_credentials():
                click.echo(
                    "Authentication required for detailed model information.", err=True
                )
                click.echo(
                    "Run 'llm github_copilot auth login' or set $GH_COPILOT_TOKEN/$GITHUB_COPILOT_TOKEN.",
                    err=True,
                )
                # Explicitly return non-zero exit code for error
                ctx = click.get_current_context()
                ctx.exit(1)

            # Only print fetching message if not raw
            if not raw:
                click.echo("Fetching detailed model information...")
            try:
                models_data = fetch_models_data(authenticator)
                api_models_info = {
                    model["id"]: model for model in models_data.get("data", [])
                }
            except Exception as e:
                click.echo(f"Error fetching model details from API: {str(e)}", err=True)
                # If raw was requested, we cannot fulfill it, so exit with error
                if raw:
                    ctx = click.get_current_context()
                    ctx.exit(1)
                # Otherwise, show basic list and exit with error
                click.echo("Showing basic registered model list instead:")
                for model_id in github_model_ids:
                    click.echo(f"- {model_id}")
                # Explicitly return non-zero exit code for error
                ctx = click.get_current_context()
                ctx.exit(1)

            # Raw output (--raw)
            if raw:
                # Print the raw JSON response, pretty-printed
                click.echo(json.dumps(models_data, indent=2))
                return  # Implicit exit 0

            # Verbose output (-v)
            elif verbose:
                click.echo("Registered GitHub Copilot models (Verbose):")

                for i, model_id in enumerate(github_model_ids):
                    _, api_model_name = model_id.split("/", 1)
                    vendor = "N/A"
                    name = "N/A"
                    version = "N/A"
                    context_length = "N/A"
                    family = "N/A"  # Initialize family

                    model_info = api_models_info[api_model_name]
                    vendor = model_info.get("vendor", "N/A")
                    name = model_info.get("name", "N/A")
                    version = model_info.get("version", "N/A")
                    # Get capabilities, limits, and supports
                    capabilities = model_info.get("capabilities", {})
                    limits = capabilities.get("limits", {})
                    supports = capabilities.get("supports", {})
                    # Extract details
                    context_length = limits.get("max_context_window_tokens", "N/A")
                    family = capabilities.get("family", "N/A")  # Get family
                    can_stream=supports.get("streaming", False)
                    supports_tools=supports.get("tool_calls", False)
                    supports_schema=supports.get("structured_outputs", False)


                    # Print model details
                    formatted_context_length = (
                        f"{context_length:,}"
                        if isinstance(context_length, int)
                        else context_length
                    )
                    click.echo(f"- id: {model_id}")
                    click.echo(f"  vendor: {vendor}")
                    click.echo(f"  name: {name}")
                    click.echo(f"  version: {version}")
                    click.echo(f"  family: {family}")
                    click.echo(f"  context_length: {formatted_context_length}")
                    click.echo(f"  streaming: {can_stream}")
                    click.echo(f"  tool calls: {supports_tools}")
                    click.echo(f"  schema (structured outputs): {supports_schema}")
                    # Format context_length with commas if it's a number

                    # Add a blank line separator between models, but not after the last one
                    if i < len(github_model_ids) - 1:
                        click.echo()
                # Successful verbose output, return normally
                return  # Implicit exit 0

        except Exception as e:
            # Make sure not to catch click.exceptions.Exit
            if isinstance(e, click.exceptions.Exit):
                raise e
            click.echo(f"Error listing registered models: {str(e)}", err=True)
            # Explicitly return non-zero exit code for error
            ctx = click.get_current_context()
            ctx.exit(1)
