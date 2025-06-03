import json
import llm
import os
import pytest
from unittest.mock import patch, MagicMock
import llm_github_copilot
from llm_github_copilot import GitHubCopilot, GitHubCopilotAuthenticator
import httpx

# Mock API key for testing
GITHUB_COPILOT_TOKEN = (
    os.environ.get("PYTEST_GITHUB_COPILOT_TOKEN", None) or "ghu_mocktoken"
)

# Mock response data
MOCK_RESPONSE_TEXT = "1. Captain\n2. Splash"


@pytest.mark.vcr
def test_prompt():
    """Test basic prompt functionality"""
    model = llm.get_model("github_copilot")

    # Mock the authenticator to avoid actual API calls
    with patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.get_api_key",
        return_value=GITHUB_COPILOT_TOKEN,
    ):
        # Mock the execute method directly
        with patch.object(model, "execute", return_value=iter([MOCK_RESPONSE_TEXT])):
            # Test the prompt
            response = model.prompt("Two names for a pet pelican, be brief")
            assert str(response) == MOCK_RESPONSE_TEXT


@pytest.mark.vcr
def test_model_variants():
    """Test that model variants are properly registered"""
    # Test that the default model exists
    default_model = llm.get_model("github_copilot")
    assert default_model is not None
    assert default_model.model_id == "github_copilot"

    # Test a variant model if it exists
    with patch(
        "llm_github_copilot.fetch_available_models",
        return_value={"github_copilot", "github_copilot/o3-mini"},
    ), patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.has_valid_credentials",
        return_value=True,
    ):
        # Re-register models to pick up our mocked variants
        for hook in llm.get_plugins():
            if hasattr(hook, "register_models"):
                hook.register_models(llm.register_model)

        variant_model = llm.get_model("github_copilot/o3-mini")
        assert variant_model is not None
        assert variant_model.model_id == "github_copilot/o3-mini"


@pytest.mark.vcr
def test_streaming_response():
    """Test streaming response functionality"""
    model = llm.get_model("github_copilot")

    # Mock the authenticator to avoid actual API calls
    with patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.get_api_key",
        return_value=GITHUB_COPILOT_TOKEN,
    ):
        # Mock the _handle_streaming_request method
        with patch.object(
            GitHubCopilot,
            "_handle_streaming_request",
            return_value=iter([MOCK_RESPONSE_TEXT]),
        ):
            # Test streaming response
            response = model.prompt(
                "Two names for a pet pelican, be brief", stream=True
            )
            assert str(response) == MOCK_RESPONSE_TEXT


@pytest.mark.vcr
def test_options():
    """Test that options are properly passed to the API"""
    model = llm.get_model("github_copilot")

    # Extract and test the options directly from the LLM prompt object
    with patch(
        "llm_github_copilot.GitHubCopilotAuthenticator.get_api_key",
        return_value=GITHUB_COPILOT_TOKEN,
    ):
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
def test_model_mappings():
    """Test the model mappings functionality"""
    # Create dummy mappings for testing
    test_mappings = {
        "github_copilot": "gpt-4o",
        "github_copilot/o3-mini": "o3-mini",
    }

    # Directly patch the class attribute with our test data
    with patch.object(
        llm_github_copilot.GitHubCopilot, "_model_mappings", test_mappings
    ):
        model = llm.get_model("github_copilot")

        # Get the mappings - should return our test data directly
        mappings = model.get_model_mappings()

        # Check the mappings match what we set
        assert "github_copilot" in mappings
        assert mappings["github_copilot"] == "gpt-4o"
        assert "github_copilot/o3-mini" in mappings


@pytest.mark.vcr
def test_get_streaming_models():
    """Test the get_streaming_models functionality"""
    # Create dummy streaming models list for testing
    test_streaming_models = ["gpt-4o", "o3-mini"]

    # Directly patch the class attribute with our test data
    with patch.object(
        llm_github_copilot.GitHubCopilot, "_streaming_models", test_streaming_models
    ):
        model = llm.get_model("github_copilot")

        # Get the streaming models - should return our test data directly
        streaming_models = model.get_streaming_models()

        # Check the streaming models match what we set
        assert "gpt-4o" in streaming_models
        assert "o3-mini" in streaming_models


@pytest.mark.vcr
def test_get_model_for_api():
    """Test the _get_model_for_api method"""
    model = llm.get_model("github_copilot")

    # Create dummy mappings for testing
    test_mappings = {
        "github_copilot": "gpt-4o",
        "github_copilot/o3-mini": "o3-mini",
    }

    # Patch the get_model_mappings method to return our test data
    with patch.object(GitHubCopilot, "get_model_mappings", return_value=test_mappings):
        # Test with default model
        api_model = model._get_model_for_api("github_copilot")
        assert api_model == "gpt-4o"

        # Test with variant model
        api_model = model._get_model_for_api("github_copilot/o3-mini")
        assert api_model == "o3-mini"

        # Test with unknown model (should return default)
        api_model = model._get_model_for_api("github_copilot/unknown-model")
        assert api_model == "gpt-4o"


@pytest.mark.vcr
def test_build_conversation_messages():
    """Test the _build_conversation_messages method"""
    model = llm.get_model("github_copilot")

    # Create a mock prompt
    mock_prompt = MagicMock()
    mock_prompt.prompt = "What is the capital of France?"

    # Test with no conversation
    messages = model._build_conversation_messages(mock_prompt, None)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the capital of France?"

    # Create a mock conversation with one response
    mock_conversation = MagicMock()
    mock_response = MagicMock()
    mock_response.prompt.prompt = "Hello"
    mock_response.text.return_value = "Hi there!"
    mock_conversation.responses = [mock_response]

    # Test with conversation
    messages = model._build_conversation_messages(mock_prompt, mock_conversation)
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there!"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "What is the capital of France?"


@pytest.mark.vcr
def test_non_streaming_request():
    """Test the _non_streaming_request method"""
    model = llm.get_model("github_copilot")

    # Create mock objects
    mock_prompt = MagicMock()
    mock_headers = {"Authorization": f"Bearer {GITHUB_COPILOT_TOKEN}"}
    mock_payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }

    # Mock the httpx.post method
    mock_response = MagicMock()
    mock_response.text = MOCK_RESPONSE_TEXT
    mock_response.json.return_value = {
        "choices": [{"message": {"content": MOCK_RESPONSE_TEXT}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.post", return_value=mock_response):
        # Test the method
        result = list(
            model._non_streaming_request(
                mock_prompt, mock_headers, mock_payload, "gpt-4o"
            )
        )
        assert result[0] == MOCK_RESPONSE_TEXT

    # Test with different response format
    mock_response.json.return_value = {"choices": [{"text": MOCK_RESPONSE_TEXT}]}
    with patch("httpx.post", return_value=mock_response):
        result = list(
            model._non_streaming_request(
                mock_prompt, mock_headers, mock_payload, "gpt-4o"
            )
        )
        assert result[0] == MOCK_RESPONSE_TEXT

    # Test with HTTP error
    with patch(
        "httpx.post",
        side_effect=httpx.HTTPStatusError(
            "Error", request=MagicMock(), response=mock_response
        ),
    ):
        result = list(
            model._non_streaming_request(
                mock_prompt, mock_headers, mock_payload, "gpt-4o"
            )
        )
        assert "HTTP error" in result[0]


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
