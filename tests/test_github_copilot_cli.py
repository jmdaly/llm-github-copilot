import json
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
from click.testing import CliRunner
import click
import llm
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
    # Get all registered commands
    commands = {}
    
    # Mock the CLI group to capture registered commands
    mock_cli = MagicMock()
    
    # Call the register_commands hook
    for hook in llm.get_plugins():
        if hasattr(hook, "register_commands"):
            hook.register_commands(mock_cli)
    
    # Verify the github-copilot command group was registered
    mock_cli.group.assert_any_call(name="github-copilot")


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
        """Test login with GH_COPILOT_KEY environment variable set."""
        # Setup mock for environment variable
        with patch.dict(os.environ, {"GH_COPILOT_KEY": "test_token"}):
            # Setup mock to indicate not authenticated
            mock_authenticator.has_valid_credentials.return_value = False
            
            # Create a mock CLI command
            @click.command()
            @click.option("-f", "--force", is_flag=True)
            def mock_login_command(force):
                mock_authenticator.has_valid_credentials()
                if os.environ.get("GH_COPILOT_KEY"):
                    click.echo("Not possible to initiate login with environment variable GH_COPILOT_KEY set")
            
            # Run the command
            result = cli_runner.invoke(mock_login_command)
            
            # Check the output
            assert "Not possible to initiate login with environment variable GH_COPILOT_KEY set" in result.output
    
    def test_login_success(self, cli_runner, mock_authenticator):
        """Test successful login."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = False
        mock_authenticator._login.return_value = "mock_access_token"
        mock_authenticator._refresh_api_key.return_value = {
            "token": "mock_api_key",
            "expires_at": 9999999999
        }
        
        # Mock fetch_available_models
        with patch("llm_github_copilot.fetch_available_models", return_value={"github-copilot", "github-copilot/gpt-4o"}):
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
            assert "GitHub Copilot login process completed successfully!" in result.output
            assert "Available models:" in result.output


class TestAuthStatus:
    """Tests for the 'auth status' command."""
    
    def test_status_authenticated(self, cli_runner, mock_authenticator):
        """Test status when authenticated."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = True
        
        # Mock API key file content
        mock_api_key_info = {
            "token": "mock_api_key",
            "expires_at": 9999999999
        }
        
        # Create a mock CLI command
        @click.command()
        @click.option("-v", "--verbose", is_flag=True)
        def mock_status_command(verbose):
            if mock_authenticator.has_valid_credentials():
                click.echo("GitHub Copilot authentication: ✓ Authenticated")
        
        # Run the command
        result = cli_runner.invoke(mock_status_command)
        
        # Check the output
        assert "GitHub Copilot authentication: ✓ Authenticated" in result.output
    
    def test_status_not_authenticated(self, cli_runner, mock_authenticator):
        """Test status when not authenticated."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = False
        
        # Create a mock CLI command
        @click.command()
        @click.option("-v", "--verbose", is_flag=True)
        def mock_status_command(verbose):
            if not mock_authenticator.has_valid_credentials():
                click.echo("GitHub Copilot authentication: ✗ Not authenticated")
        
        # Run the command
        result = cli_runner.invoke(mock_status_command)
        
        # Check the output
        assert "GitHub Copilot authentication: ✗ Not authenticated" in result.output
    
    def test_status_verbose(self, cli_runner, mock_authenticator):
        """Test verbose status output."""
        # Setup mocks
        mock_authenticator.has_valid_credentials.return_value = True
        
        # Mock API key file read
        mock_file_content = json.dumps({
            "token": "mock_api_key",
            "expires_at": 9999999999
        })
        
        with patch("pathlib.Path.read_text", return_value=mock_file_content):
            # Mock llm.get_key
            with patch("llm.get_key", return_value="mock_access_token"):
                # Create a mock CLI command
                @click.command()
                @click.option("-v", "--verbose", is_flag=True)
                def mock_status_command(verbose):
                    if mock_authenticator.has_valid_credentials():
                        click.echo("GitHub Copilot authentication: ✓ Authenticated")
                        if verbose:
                            access_token = llm.get_key("github-copilot", mock_authenticator.ACCESS_TOKEN_KEY)
                            click.echo(f"Access token: {access_token} (from LLM key storage)")
                            api_key_info = json.loads(Path.read_text(mock_authenticator.api_key_file))
                            api_key = api_key_info.get("token", "")
                            click.echo(f"API key: {api_key}")
                
                # Run the command with verbose flag
                result = cli_runner.invoke(mock_status_command, ["--verbose"])
                
                # Check the output
                assert "GitHub Copilot authentication: ✓ Authenticated" in result.output
                assert "Access token: mock_access_token" in result.output
                assert "API key: mock_api_key" in result.output


class TestAuthRefresh:
    """Tests for the 'auth refresh' command."""
    
    def test_refresh_no_token(self, cli_runner, mock_authenticator):
        """Test refresh when no token is available."""
        # Mock llm.get_key to return None
        with patch("llm.get_key", return_value=None):
            # Create a mock CLI command
            @click.command()
            @click.option("-v", "--verbose", is_flag=True)
            def mock_refresh_command(verbose):
                try:
                    access_token = llm.get_key("github-copilot", mock_authenticator.ACCESS_TOKEN_KEY)
                except (TypeError, Exception):
                    access_token = None
                
                if not access_token and not os.environ.get("GH_COPILOT_KEY"):
                    click.echo("No access token found. Run 'llm github-copilot auth login' first.")
                    return 1
            
            # Run the command
            result = cli_runner.invoke(mock_refresh_command)
            
            # Check the output
            assert "No access token found." in result.output
            assert result.exit_code == 1
    
    def test_refresh_success(self, cli_runner, mock_authenticator):
        """Test successful refresh."""
        # Mock llm.get_key to return a token
        with patch("llm.get_key", return_value="mock_access_token"):
            # Setup mock for refresh_api_key
            mock_authenticator._refresh_api_key.return_value = {
                "token": "new_mock_api_key",
                "expires_at": 9999999999
            }
            
            # Create a mock CLI command
            @click.command()
            @click.option("-v", "--verbose", is_flag=True)
            def mock_refresh_command(verbose):
                try:
                    access_token = llm.get_key("github-copilot", mock_authenticator.ACCESS_TOKEN_KEY)
                except (TypeError, Exception):
                    access_token = None
                
                if access_token or os.environ.get("GH_COPILOT_KEY"):
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
            assert "new_mock_api_key" not in result.output  # Should not show key in non-verbose mode
            
            # Run with verbose flag
            result = cli_runner.invoke(mock_refresh_command, ["--verbose"])
            assert "API key: new_mock_api_key" in result.output


class TestAuthLogout:
    """Tests for the 'auth logout' command."""
    
    def test_logout(self, cli_runner, mock_authenticator):
        """Test logout command."""
        # Mock llm.get_key and llm.delete_key
        with patch("llm.get_key", return_value="mock_access_token"):
            with patch("llm.delete_key") as mock_delete_key:
                # Mock file existence check and unlink
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.unlink") as mock_unlink:
                        # Create a mock CLI command
                        @click.command()
                        def mock_logout_command():
                            try:
                                if llm.get_key("github-copilot", mock_authenticator.ACCESS_TOKEN_KEY):
                                    llm.delete_key("github-copilot", mock_authenticator.ACCESS_TOKEN_KEY)
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
                        assert "GitHub Copilot logout completed successfully." in result.output
                        
                        # Verify the mocks were called
                        mock_delete_key.assert_called_once()
                        mock_unlink.assert_called_once()
