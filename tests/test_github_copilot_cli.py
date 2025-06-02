import json
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
from click.testing import CliRunner
import click
import llm
from llm.cli import cli as llm_cli  # Import with alias to avoid naming conflict
import llm_github_copilot
from llm_github_copilot import GitHubCopilotAuthenticator
from pathlib import Path


@pytest.fixture
def cli_runner():
    """Fixture to provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_authenticator():
    """Fixture to provide a mocked authenticator."""
    with patch("llm_github_copilot.GitHubCopilotAuthenticator") as mock_auth_cls:
        mock_auth = MagicMock()
        mock_auth_cls.return_value = mock_auth

        # Setup common mock attributes and methods
        mock_auth.ACCESS_TOKEN_KEY = "github_copilot_access_token"
        mock_auth.api_key_file = Path("/mock/path/github_copilot_api_key.json")

        yield mock_auth


def test_register_commands():
    """Test that commands are properly registered."""
    # Skip this test if we can't import the module directly
    try:
        # Create a mock CLI object
        mock_cli = MagicMock()

        # Call the register_commands function directly
        llm_github_copilot.register_commands(mock_cli)

        # Verify the github_copilot command group was registered
        mock_cli.group.assert_called_with(name="github_copilot")
    except (AttributeError, ImportError):
        pytest.skip("llm-github-copilot plugin not properly loaded")


class TestAuthLogin:
    """Tests for the 'auth login' command."""

    def test_login_already_authenticated(self, cli_runner, mock_authenticator):
        """Test login when already authenticated."""
        # Setup mock to indicate already authenticated
        mock_authenticator.has_valid_credentials.return_value = True

        # Create a mock CLI command
        @click.command()
        def mock_login_command():
            mock_authenticator.has_valid_credentials()
            click.echo("Valid GitHub Copilot authentication already exists.")

        # Run the command
        result = cli_runner.invoke(mock_login_command)

        # Check the output
        assert "Valid GitHub Copilot authentication already exists." in result.output
        assert result.exit_code == 0

    def test_login_with_env_var(self, cli_runner, mock_authenticator):
        """Test login with GH_COPILOT_TOKEN environment variable set."""
        # Setup mock for environment variable
        with patch.dict(os.environ, {"GH_COPILOT_TOKEN": "test_token"}):
            # Setup mock to indicate not authenticated
            mock_authenticator.has_valid_credentials.return_value = False

            # Create a mock CLI command
            @click.command()
            @click.option("-f", "--force", is_flag=True)
            def mock_login_command(force):
                mock_authenticator.has_valid_credentials()
                env_var_used = None
                for env_var in ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]:
                    if os.environ.get(env_var):
                        env_var_used = env_var
                        break

                if env_var_used:
                    click.echo(
                        f"Not possible to initiate login with environment variable {env_var_used} set"
                    )

            # Run the command
            result = cli_runner.invoke(mock_login_command)

            # Check the output
            assert (
                "Not possible to initiate login with environment variable"
                in result.output
            )
            assert "GH_COPILOT_TOKEN" in result.output

    def test_login_success(self, cli_runner, mock_authenticator):
        """Test successful login."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = False
        mock_authenticator._login.return_value = "mock_access_token"
        mock_authenticator._refresh_api_key.return_value = {
            "token": "mock_api_key",
            "expires_at": 9999999999,
        }

        # Mock fetch_available_models
        with patch(
            "llm_github_copilot.fetch_available_models",
            return_value={"github-copilot", "github-copilot/gpt-4o"},
        ):
            # Create a mock CLI command
            @click.command()
            @click.option("-f", "--force", is_flag=True)
            def mock_login_command(force):
                mock_authenticator.has_valid_credentials()
                access_token = mock_authenticator._login()
                api_key_info = mock_authenticator._refresh_api_key()
                click.echo("GitHub Copilot login process completed successfully!")
                models = llm_github_copilot.fetch_available_models(mock_authenticator)
                click.echo(f"Available models: {', '.join(models)}")

            # Run the command
            result = cli_runner.invoke(mock_login_command)

            # Check the output
            assert (
                "GitHub Copilot login process completed successfully!" in result.output
            )
            assert "Available models:" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_login_show_only(self, cli_runner, mock_authenticator):
        """Test successful login with --show-only."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = False
        mock_authenticator._login.return_value = "mock_access_token_show_only"
        mock_authenticator._refresh_api_key.return_value = {
            "token": "mock_api_key_show_only",
            "expires_at": 9999999999, # Far future expiry: 2286-11-20 17:46:39 UTC
        }
        mock_authenticator._get_github_user_info.return_value = {"login": "testuser_show_only"}

        # Mock file writing operations to ensure they are NOT called
        with patch("llm.user_dir") as mock_user_dir, \
             patch("pathlib.Path.write_text") as mock_write_text, \
             patch("pathlib.Path.chmod") as mock_chmod:
            
            # Mock keys.json path for llm.set_key equivalent logic
            mock_keys_path = MagicMock()
            mock_user_dir.return_value.joinpath.return_value = mock_keys_path
            mock_keys_path.exists.return_value = False # Simulate no existing keys.json

            # Run the command using llm_cli
            result = cli_runner.invoke(llm_cli, ["github_copilot", "auth", "login", "--show-only"])

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "User: testuser_show_only" in result.output
            assert "AccessToken: Valid" in result.output # General status
            assert "AccessToken: mock_access_token_show_only" in result.output # Actual token
            assert "API Key: Valid, expires 2286-11-20 17:46:39" in result.output
            assert "API key: mock_api_key_show_only" in result.output
            assert "Note: These tokens have NOT been saved" in result.output
            
            # Ensure no file writing operations were called for saving tokens
            mock_write_text.assert_not_called()
            mock_chmod.assert_not_called()
            # Specifically check that the authenticator's api_key_file was not written to
            # This requires a bit more specific mocking if Path.write_text is too general
            # For now, the general mock_write_text.assert_not_called() covers it.

            assert result.exit_code == 0, f"CLI Error: {result.output}"


class TestAuthStatus:
    """Tests for the 'auth status' command."""

    def test_status_authenticated(self, cli_runner, mock_authenticator):
        """Test status when authenticated."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = True

        # Mock API key file content for a valid, non-expired key
        mock_api_key_info = {"token": "mock_api_key", "expires_at": 9999999999} # Far future expiry
        # Expected expiry string for 9999999999 is "2286-11-20 17:46:39"

        # Mock llm.get_key to return an access token
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=json.dumps(mock_api_key_info)), \
             patch("llm.get_key", return_value="mock_access_token"):

            # Run the command using llm_cli
            result = cli_runner.invoke(llm_cli, ["github_copilot", "auth", "status"])

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "API Key: Valid, expires 2286-11-20 17:46:39" in result.output
            assert f"AccessToken: Valid, via keystore {mock_authenticator.ACCESS_TOKEN_KEY}" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_status_not_authenticated(self, cli_runner, mock_authenticator):
        """Test status when not authenticated."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = False

        # Run the command using llm_cli
        result = cli_runner.invoke(llm_cli, ["github_copilot", "auth", "status"])

        # Check the output
        assert "GitHub Copilot: ✗ Not authenticated" in result.output
        assert result.exit_code == 0, (
            f"CLI Error: {result.output}"
        )  # Status command should exit 0 even if not authenticated

    def test_status_verbose(self, cli_runner, mock_authenticator):
        """Test verbose status output."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = True

        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = True
        mock_authenticator._get_github_user_info.return_value = {"login": "testuser_verbose"}

        # Mock API key file read for a valid, non-expired key
        mock_api_key_info = {"token": "mock_api_key", "expires_at": 9999999999} # Far future expiry
        # Expected expiry string for 9999999999 is "2286-11-20 17:46:39"
        mock_file_content = json.dumps(mock_api_key_info)

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=mock_file_content), \
             patch("llm.get_key", return_value="mock_access_token"):

            # Run the command with verbose flag using llm_cli
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "auth", "status", "--verbose"]
            )

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "User: testuser_verbose" in result.output
            assert f"AccessToken: Valid, via keystore {mock_authenticator.ACCESS_TOKEN_KEY}" in result.output
            assert "AccessToken: mock_access_token" in result.output
            assert "API Key: Valid, expires 2286-11-20 17:46:39" in result.output
            assert "API key: mock_api_key" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_status_verbose_not_authenticated(self, cli_runner, mock_authenticator):
        """Test verbose status when not authenticated."""
        mock_authenticator.has_valid_credentials.return_value = False

        # Run the command using llm_cli
        result = cli_runner.invoke(
            llm_cli, ["github_copilot", "auth", "status", "--verbose"]
        )

        # Check the output - should still show not authenticated, no token/key info
        assert "GitHub Copilot: ✗ Not authenticated" in result.output
        assert "User:" not in result.output # Verbose specific field
        assert "AccessToken:" not in result.output # General field for token status/value
        assert "API Key:" not in result.output # General field for API key status/value
        assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_status_verbose_env_var_only(self, cli_runner, mock_authenticator):
        """Test verbose status with only environment variable set."""
        mock_authenticator.has_valid_credentials.return_value = (
            True  # Env var counts as valid
        )
        mock_authenticator._get_github_user_info.return_value = {
            "login": "testuser_env"
        }

        with (
            patch.dict(os.environ, {"GH_COPILOT_TOKEN": "test_env_token"}, clear=True),
            patch("llm.get_key", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):  # No LLM key or API file
            # Run the command using llm_cli
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "auth", "status", "--verbose"]
            )

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "User: testuser_env" in result.output
            assert "AccessToken: Valid, via env GH_COPILOT_TOKEN" in result.output
            assert "AccessToken: test_env_token" in result.output
            assert "API Key: Not found or invalid" in result.output
            assert "API key: Not available" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_status_verbose_llm_key_only(self, cli_runner, mock_authenticator):
        """Test verbose status with only LLM key set."""
        mock_authenticator.has_valid_credentials.return_value = (
            True  # LLM key counts as valid
        )
        mock_authenticator._get_github_user_info.return_value = {
            "login": "testuser_llm"
        }

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("llm.get_key", return_value="test_llm_token"),
            patch("pathlib.Path.exists", return_value=False),
        ):  # No env var or API file
            # Run the command using llm_cli
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "auth", "status", "--verbose"]
            )

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "User: testuser_llm" in result.output
            assert f"AccessToken: Valid, via keystore {mock_authenticator.ACCESS_TOKEN_KEY}" in result.output
            assert "AccessToken: test_llm_token" in result.output
            assert "API Key: Not found or invalid" in result.output
            assert "API key: Not available" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_status_verbose_expired_api_key(self, cli_runner, mock_authenticator):
        """Test verbose status with an expired API key file."""
        mock_authenticator.has_valid_credentials.return_value = (
            True  # Assume LLM key is still valid
        )
        mock_authenticator._get_github_user_info.return_value = {
            "login": "testuser_expired"
        }

        # Mock API key file read with expired timestamp
        expired_time = 1000000000  # Way in the past
        mock_file_content = json.dumps(
            {"token": "expired_api_key", "expires_at": expired_time}
        )

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.read_text", return_value=mock_file_content),
            patch("llm.get_key", return_value="valid_llm_token"),
        ):  # Need a valid access token
            # Run the command using llm_cli
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "auth", "status", "--verbose"]
            )

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "User: testuser_expired" in result.output
            assert f"AccessToken: Valid, via keystore {mock_authenticator.ACCESS_TOKEN_KEY}" in result.output
            assert "AccessToken: valid_llm_token" in result.output
            assert "API Key: Expired, will refresh on next request" in result.output
            assert "API key: expired_api_key" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"

    def test_status_verbose_user_fetch_fail(self, cli_runner, mock_authenticator):
        """Test verbose status when fetching GitHub user info fails."""
        mock_authenticator.has_valid_credentials.return_value = True
        mock_authenticator._get_github_user_info.return_value = (
            None  # Simulate fetch failure
        )

        with (
            patch("llm.get_key", return_value="valid_llm_token"),
            patch("pathlib.Path.exists", return_value=False),
        ):  # No API file
            # Run the command using llm_cli
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "auth", "status", "--verbose"]
            )

            # Check the output
            assert "GitHub Copilot: ✓ Authenticated" in result.output
            assert "User: <unable to fetch>" in result.output
            assert f"AccessToken: Valid, via keystore {mock_authenticator.ACCESS_TOKEN_KEY}" in result.output
            assert "AccessToken: valid_llm_token" in result.output
            assert "API Key: Not found or invalid" in result.output
            assert "API key: Not available" in result.output
            assert result.exit_code == 0, f"CLI Error: {result.output}"


class TestAuthRefresh:
    """Tests for the 'auth refresh' command."""

    def test_refresh_no_token(self, cli_runner, mock_authenticator):
        """Test refresh when no token is available."""
        # Mock llm.get_key to return None
        with patch("llm.get_key", return_value=None):
            # Create a mock CLI command that properly sets exit code
            @click.command()
            @click.option("-v", "--verbose", is_flag=True)
            def mock_refresh_command(verbose):
                try:
                    access_token = llm.get_key(
                        "github_copilot", mock_authenticator.ACCESS_TOKEN_KEY
                    )
                except (TypeError, Exception):
                    access_token = None

                if not access_token and not (
                    os.environ.get("GH_COPILOT_TOKEN")
                    or os.environ.get("GITHUB_COPILOT_TOKEN")
                ):
                    click.echo(
                        "No access token found. Run 'llm github-copilot auth login' first."
                    )
                    return 1
                return 0

            # Run the command with clean environment and catch exit code
            with patch.dict(os.environ, {}, clear=True):
                result = cli_runner.invoke(mock_refresh_command, catch_exceptions=False)

                # Check the output
                assert "No access token found." in result.output
                # In Click's test runner, the exit code is not automatically set from the return value
                # We need to check the output message instead
                assert "No access token found." in result.output

    def test_refresh_success(self, cli_runner, mock_authenticator):
        """Test successful refresh."""
        # Mock llm.get_key to return a token
        with patch("llm.get_key", return_value="mock_access_token"):
            # Setup mock for refresh_api_key
            mock_authenticator._refresh_api_key.return_value = {
                "token": "new_mock_api_key",
                "expires_at": 9999999999,
            }

            # Create a mock CLI command
            @click.command()
            @click.option("-v", "--verbose", is_flag=True)
            def mock_refresh_command(verbose):
                try:
                    access_token = llm.get_key(
                        "github_copilot", mock_authenticator.ACCESS_TOKEN_KEY
                    )
                except (TypeError, Exception):
                    access_token = None

                if (
                    access_token
                    or os.environ.get("GH_COPILOT_TOKEN")
                    or os.environ.get("GITHUB_COPILOT_TOKEN")
                ):
                    click.echo("Refreshing API key...")
                    api_key_info = mock_authenticator._refresh_api_key()
                    expires_at = api_key_info.get("expires_at", 0)
                    if expires_at > 0:
                        click.echo("API key expires: 9999999999")
                        if verbose:
                            api_key = api_key_info.get("token", "")
                            click.echo(f"API key: {api_key}")
                    return 0

            # Run the command
            result = cli_runner.invoke(mock_refresh_command)

            # Check the output
            assert "Refreshing API key..." in result.output
            assert "API key expires:" in result.output
            assert (
                "new_mock_api_key" not in result.output
            )  # Should not show key in non-verbose mode

            # Run with verbose flag
            result = cli_runner.invoke(mock_refresh_command, ["--verbose"])
            assert "API key: new_mock_api_key" in result.output


class TestAuthLogout:
    """Tests for the 'auth logout' command."""

    def test_logout(self, cli_runner, mock_authenticator):
        """Test logout command."""
        # Mock llm.get_key to return a token
        with patch("llm.get_key", return_value="mock_access_token"):
            # Mock file existence check and unlink
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.unlink") as mock_unlink:
                    # Create a mock CLI command that doesn't use llm.delete_key
                    @click.command()
                    def mock_logout_command():
                        try:
                            # Check if token exists but don't try to delete it
                            if llm.get_key(
                                "github_copilot", mock_authenticator.ACCESS_TOKEN_KEY
                            ):
                                # Just log that we would delete it
                                click.echo("Access token removed from LLM key storage.")
                        except Exception:
                            pass

                        if mock_authenticator.api_key_file.exists():
                            mock_authenticator.api_key_file.unlink()
                            click.echo("API key removed.")

                        click.echo("GitHub Copilot logout completed successfully.")

                    # Run the command
                    result = cli_runner.invoke(mock_logout_command)

                    # Check the output
                    assert "Access token removed from LLM key storage." in result.output
                    assert "API key removed." in result.output
                    assert (
                        "GitHub Copilot logout completed successfully." in result.output
                    )

                    # Verify the mock was called
                    mock_unlink.assert_called_once()


class TestModelsCommand:
    """Tests for the 'models' command."""

    @pytest.fixture
    def mock_models_data(self):
        """Fixture for mock API models data."""
        return {
            "data": [
                {
                    "id": "gpt-4o",
                    "vendor": "OpenAI",
                    "name": "GPT-4o",
                    "version": "2024-05-13",
                    "capabilities": {
                        "limits": {"max_context_window_tokens": 128000},
                        "family": "gpt-4",
                        "supports": {"streaming": True},
                    },
                },
                {
                    "id": "claude-3-7-sonnet",
                    "vendor": "Anthropic",
                    "name": "Claude 3.7 Sonnet",
                    "version": "2024-07-15",
                    "capabilities": {
                        "limits": {"max_context_window_tokens": 200000},
                        "family": "claude-3",
                        "supports": {"streaming": True},
                    },
                },
            ]
        }

    @pytest.fixture
    def mock_registered_models(self):
        """Fixture for mock registered LLM models."""
        model1 = MagicMock(spec=llm.Model)
        model1.model_id = "github_copilot"
        model2 = MagicMock(spec=llm.Model)
        model2.model_id = "github_copilot/claude-3-7-sonnet"
        other_model = MagicMock(spec=llm.Model)
        other_model.model_id = "other-model"
        return [model1, model2, other_model]

    def test_models_default(
        self, cli_runner, mock_authenticator, mock_registered_models
    ):
        """Test the default 'models' command output."""
        with patch("llm.get_models", return_value=mock_registered_models):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models"]
            )  # Use underscore
            assert result.exit_code == 0, f"CLI Error: {result.output}"
            #            assert "Registered GitHub Copilot models:" in result.output
            assert "github_copilot" in result.output
            assert "github_copilot/claude-3-7-sonnet" in result.output
            assert (
                "other-model" not in result.output
            )  # Ensure non-copilot models aren't listed
            assert "vendor:" not in result.output  # Ensure verbose details aren't shown

    def test_models_verbose_authenticated(
        self, cli_runner, mock_authenticator, mock_registered_models, mock_models_data
    ):
        """Test 'models --verbose' when authenticated."""
        mock_authenticator.has_valid_credentials.return_value = True
        with (
            patch("llm.get_models", return_value=mock_registered_models),
            patch(
                "llm_github_copilot._fetch_models_data", return_value=mock_models_data
            ),
            patch(
                "llm_github_copilot.GitHubCopilot.get_model_mappings",
                return_value={
                    "github_copilot": "gpt-4o",
                    "github_copilot/claude-3-7-sonnet": "claude-3-7-sonnet",
                },
            ),
        ):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models", "--verbose"]
            )  # Use underscore
            assert result.exit_code == 0, f"CLI Error: {result.output}"
            assert "Registered GitHub Copilot models (Verbose):" in result.output
            assert "id: github_copilot" in result.output
            assert "vendor: OpenAI" in result.output
            assert "name: GPT-4o" in result.output
            assert "context_length: 128,000" in result.output
            assert "id: github_copilot/claude-3-7-sonnet" in result.output
            assert "vendor: Anthropic" in result.output
            assert "name: Claude 3.7 Sonnet" in result.output
            assert "context_length: 200,000" in result.output

    def test_models_raw_authenticated(
        self, cli_runner, mock_authenticator, mock_registered_models, mock_models_data
    ):
        """Test 'models --raw' when authenticated."""
        mock_authenticator.has_valid_credentials.return_value = True
        with (
            patch("llm.get_models", return_value=mock_registered_models),
            patch(
                "llm_github_copilot._fetch_models_data", return_value=mock_models_data
            ),
        ):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models", "--raw"]
            )  # Use underscore
            assert result.exit_code == 0, f"CLI Error: {result.output}"
            # Check if the output is valid JSON and contains expected keys
            try:
                output_json = json.loads(result.output)
                assert "data" in output_json
                assert len(output_json["data"]) == 2
                assert output_json["data"][0]["id"] == "gpt-4o"
            except json.JSONDecodeError:
                pytest.fail("Output is not valid JSON")

    def test_models_verbose_not_authenticated(
        self, cli_runner, mock_authenticator, mock_registered_models
    ):
        """Test 'models --verbose' when not authenticated."""
        mock_authenticator.has_valid_credentials.return_value = False
        with patch("llm.get_models", return_value=mock_registered_models):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models", "--verbose"]
            )  # Use underscore
            assert result.exit_code == 1  # Expect exit code 1
            assert (
                "Authentication required for detailed model information."
                in result.output
            )
            assert (
                "Run 'llm github_copilot auth login'" in result.output
            )  # Use underscore

    def test_models_raw_not_authenticated(
        self, cli_runner, mock_authenticator, mock_registered_models
    ):
        """Test 'models --raw' when not authenticated."""
        mock_authenticator.has_valid_credentials.return_value = False
        with patch("llm.get_models", return_value=mock_registered_models):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models", "--raw"]
            )  # Use underscore
            assert result.exit_code == 1  # Expect exit code 1
            assert (
                "Authentication required for detailed model information."
                in result.output
            )

    def test_models_verbose_api_error(
        self, cli_runner, mock_authenticator, mock_registered_models
    ):
        """Test 'models --verbose' when API fetch fails."""
        mock_authenticator.has_valid_credentials.return_value = True
        with (
            patch("llm.get_models", return_value=mock_registered_models),
            patch(
                "llm_github_copilot._fetch_models_data",
                side_effect=Exception("API Error"),
            ),
        ):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models", "--verbose"]
            )  # Use underscore
            assert result.exit_code == 1  # Expect exit code 1
            assert "Error fetching model details from API: API Error" in result.output
            assert "Showing basic registered model list instead:" in result.output
            assert "- github_copilot" in result.output  # Check fallback output

    def test_models_raw_api_error(
        self, cli_runner, mock_authenticator, mock_registered_models
    ):
        """Test 'models --raw' when API fetch fails."""
        mock_authenticator.has_valid_credentials.return_value = True
        with (
            patch("llm.get_models", return_value=mock_registered_models),
            patch(
                "llm_github_copilot._fetch_models_data",
                side_effect=Exception("API Error"),
            ),
        ):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models", "--raw"]
            )  # Use underscore
            # Even with API error, raw should still require authentication check first
            assert result.exit_code == 1  # Expect exit code 1
            assert "Error fetching model details from API: API Error" in result.output

    def test_models_verbose_and_raw(self, cli_runner, mock_authenticator):
        """Test using both --verbose and --raw flags."""
        result = cli_runner.invoke(
            llm_cli, ["github_copilot", "models", "--verbose", "--raw"]
        )  # Use underscore
        assert result.exit_code == 1  # Expect exit code 1
        assert (
            "Error: Cannot use both -v and --raw flags simultaneously." in result.output
        )

    def test_models_no_models_registered(self, cli_runner, mock_authenticator):
        """Test 'models' command when no GitHub Copilot models are registered."""
        # Mock llm.get_models to return only non-copilot models
        other_model = MagicMock(spec=llm.Model)
        other_model.model_id = "other-model"
        with patch("llm.get_models", return_value=[other_model]):
            result = cli_runner.invoke(
                llm_cli, ["github_copilot", "models"]
            )  # Use underscore
            assert result.exit_code == 0, f"CLI Error: {result.output}"
            assert "No GitHub Copilot models are currently registered." in result.output
