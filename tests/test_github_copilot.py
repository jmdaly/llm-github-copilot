import json
import llm
import os
import pytest
from unittest.mock import patch, MagicMock
import llm_github_copilot
from llm_github_copilot import GitHubCopilotChat, GitHubCopilotAuthenticator
import httpx

DEFAULT_MODEL = "github_copilot/gpt-4.1"

# Mock API key for testing
GITHUB_COPILOT_TOKEN = (
    os.environ.get("PYTEST_GITHUB_COPILOT_TOKEN", None) or "ghu_mocktoken"
)

# Mock response data
MOCK_RESPONSE_TEXT = "1. Captain\n2. Splash"

MOCK_MODEL_DATA = {
    "data": [
        {
            "id": "gpt-4.1",
            "name": "GPT-4.1",
            "version": "gpt-4.1-2025-04-14"
        },
        {
            "id": "o3-mini",
            "name": "o3-mini",
            "version": "o3-mini-2025-01-31"
        }
    ]
}

@pytest.mark.vcr
def test_prompt():
    """Test basic prompt functionality"""
    # Mock the authenticator to avoid actual API calls
    with patch(
        "llm_github_copilot.fetch_models_data",
        return_value=MOCK_MODEL_DATA,
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.has_valid_credentials",
        return_value=True,
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.get_api_key",
        return_value=GITHUB_COPILOT_TOKEN,
    ):
        model = llm.get_model(DEFAULT_MODEL)
        # Mock the execute method directly
        with patch.object(model, "execute", return_value=iter([MOCK_RESPONSE_TEXT])):
            # Test the prompt
            response = model.prompt("Two names for a pet pelican, be brief")
            assert str(response) == MOCK_RESPONSE_TEXT


@pytest.mark.vcr
def test_model_variants():
    """Test that model variants are properly registered"""
    # Test a variant model if it exists
    with patch(
        "llm_github_copilot.fetch_models_data",
        return_value=MOCK_MODEL_DATA,
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.has_valid_credentials",
        return_value=True,
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.get_api_key",
        return_value=GITHUB_COPILOT_TOKEN,
    ):
        # Test that the default model exists
        default_model = llm.get_model(DEFAULT_MODEL)
        assert default_model is not None
        assert default_model.model_id == DEFAULT_MODEL

        # Re-register models to pick up our mocked variants
        for hook in llm.get_plugins():
            if hasattr(hook, "register_models"):
                hook.register_models(llm.register_model)

        variant_model = llm.get_model("github_copilot/o3-mini")
        assert variant_model is not None
        assert variant_model.model_id == "github_copilot/o3-mini"


@pytest.mark.vcr
def test_options():
    """Test that options are properly passed to the API"""
    # Extract and test the options directly from the LLM prompt object
    with patch(
        "llm_github_copilot.fetch_models_data",
        return_value=MOCK_MODEL_DATA,
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.has_valid_credentials",
        return_value=True,
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.get_api_key",
        return_value=GITHUB_COPILOT_TOKEN,
    ):
        model = llm.get_model(DEFAULT_MODEL)
        # Create a function to return our mock response but also capture the call args
        def mock_response_generator(*args, **kwargs):
            return iter([MOCK_RESPONSE_TEXT])

        # We need to patch the model's execute method
        with patch.object(model, "execute", return_value=iter([MOCK_RESPONSE_TEXT])):
            # Test with custom options
            response = model.prompt(
                "Two names for a pet pelican, be brief", max_tokens=100, temperature=0.7
            )

            # The options are directly available on the response's prompt object
            assert response.prompt.options is not None
            assert response.prompt.options.max_tokens == 100
            assert response.prompt.options.temperature == 0.7


@pytest.mark.vcr
def test_authenticator(tmp_path):
    """Test the authenticator functionality"""
    # Create a clean authenticator for testing
    authenticator = llm_github_copilot.GitHubCopilotAuthenticator()

    mock_auth_file = tmp_path / "auth.json"
    mock_data = {"token": GITHUB_COPILOT_TOKEN, "expires_at": 9999999999}
    mock_auth_file.write_text(json.dumps(mock_data))
    authenticator.api_key_file = mock_auth_file

    # Now get the API key - should read from the "file"
    api_key = authenticator.get_api_key()

    # Verify we got the expected token
    assert api_key == GITHUB_COPILOT_TOKEN


@pytest.mark.vcr
def test_authenticator_has_valid_credentials():
    """Test the has_valid_credentials method of the authenticator"""
    authenticator = GitHubCopilotAuthenticator()

    # Test with valid API key file
    with patch("pathlib.Path.exists", return_value=True):
        with patch(
            "pathlib.Path.read_text",
            return_value=json.dumps(
                {"token": GITHUB_COPILOT_TOKEN, "expires_at": 9999999999}
            ),
        ):
            assert authenticator.has_valid_credentials() is True

    # Test with expired API key file
    with patch("pathlib.Path.exists", return_value=True):
        with patch(
            "pathlib.Path.read_text",
            return_value=json.dumps(
                {"token": GITHUB_COPILOT_TOKEN, "expires_at": 1000000000}
            ),
        ):
            # Also mock llm.get_key to return a valid token
            with patch("llm.get_key", return_value="valid_token"):
                assert authenticator.has_valid_credentials() is True

    # Test with no valid credentials
    with patch("pathlib.Path.exists", return_value=False):
        with patch("llm.get_key", return_value=None):
            assert authenticator.has_valid_credentials() is False


@pytest.mark.vcr
def test_authenticator_get_access_token():
    """Test the get_access_token method of the authenticator"""
    authenticator = GitHubCopilotAuthenticator()

    # Test with environment variable
    with patch.dict(os.environ, {"GH_COPILOT_TOKEN": "env_token"}):
        assert authenticator.get_access_token() == "env_token"

    # Test with LLM key storage
    with patch.dict(os.environ, {}, clear=True):  # Clear env vars
        with patch("llm.get_key", return_value="stored_token"):
            assert authenticator.get_access_token() == "stored_token"

    # Test with no valid token (should raise exception)
    with patch.dict(os.environ, {}, clear=True):  # Clear env vars
        with patch("llm.get_key", return_value=None):
            with pytest.raises(Exception) as excinfo:
                authenticator.get_access_token()
            assert "GitHub Copilot authentication required" in str(excinfo.value)
            assert "llm github_copilot auth login" in str(excinfo.value)
