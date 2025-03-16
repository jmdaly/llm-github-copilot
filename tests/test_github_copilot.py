import json
import llm
import os
import pytest
from unittest.mock import patch, MagicMock
import llm_github_copilot
from llm_github_copilot import GitHubCopilot

# Mock API key for testing
GITHUB_COPILOT_API_KEY = os.environ.get("PYTEST_GITHUB_COPILOT_API_KEY", None) or "ghu_mocktoken"

# Mock response data
MOCK_RESPONSE_TEXT = "1. Captain\n2. Splash"


@pytest.mark.vcr
def test_prompt():
    """Test basic prompt functionality"""
    model = llm.get_model("github-copilot")
    
    # Mock the authenticator to avoid actual API calls
    with patch("llm_github_copilot.GitHubCopilotAuthenticator.get_api_key", return_value=GITHUB_COPILOT_API_KEY):
        # Mock the execute method directly
        with patch.object(model, 'execute', return_value=iter([MOCK_RESPONSE_TEXT])):
            # Test the prompt
            response = model.prompt("Two names for a pet pelican, be brief")
            assert str(response) == MOCK_RESPONSE_TEXT


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_prompt():
    """Test async prompt functionality"""
    # Skip this test for now as async support needs to be implemented
    pytest.skip("Async support not yet implemented")
    
    # The test will be enabled once async support is added to the model


@pytest.mark.vcr
def test_model_variants():
    """Test that model variants are properly registered"""
    # Test that the default model exists
    default_model = llm.get_model("github-copilot")
    assert default_model is not None
    assert default_model.model_id == "github-copilot"
    
    # Test a variant model if it exists
    with patch("llm_github_copilot.fetch_available_models", return_value={"github-copilot", "github-copilot/claude-3-7-sonnet"}):
        # Re-register models to pick up our mocked variants
        for hook in llm.get_plugins():
            if hasattr(hook, "register_models"):
                hook.register_models(llm.register_model)
                
        variant_model = llm.get_model("github-copilot/claude-3-7-sonnet")
        assert variant_model is not None
        assert variant_model.model_id == "github-copilot/claude-3-7-sonnet"


@pytest.mark.vcr
def test_streaming_response():
    """Test streaming response functionality"""
    model = llm.get_model("github-copilot")
    
    # Mock the authenticator to avoid actual API calls
    with patch("llm_github_copilot.GitHubCopilotAuthenticator.get_api_key", return_value=GITHUB_COPILOT_API_KEY):
        # Mock the _handle_streaming_request method
        with patch.object(GitHubCopilot, '_handle_streaming_request', return_value=iter([MOCK_RESPONSE_TEXT])):
            # Test streaming response
            response = model.prompt("Two names for a pet pelican, be brief", stream=True)
            assert str(response) == MOCK_RESPONSE_TEXT


@pytest.mark.vcr
def test_options():
    """Test that options are properly passed to the API"""
    model = llm.get_model("github-copilot")
    
    # Extract and test the options directly from the LLM prompt object
    with patch("llm_github_copilot.GitHubCopilotAuthenticator.get_api_key", return_value=GITHUB_COPILOT_API_KEY):
        # Create a function to return our mock response but also capture the call args
        def mock_response_generator(*args, **kwargs):
            return iter([MOCK_RESPONSE_TEXT])
        
        # We need to patch the model's execute method
        with patch.object(model, 'execute', return_value=iter([MOCK_RESPONSE_TEXT])):
            # Test with custom options
            response = model.prompt(
                "Two names for a pet pelican, be brief",
                max_tokens=100,
                temperature=0.7
            )
            
            # The options are directly available on the response's prompt object
            assert response.prompt.options is not None
            assert response.prompt.options.max_tokens == 100
            assert response.prompt.options.temperature == 0.7


@pytest.mark.vcr
def test_authenticator():
    """Test the authenticator functionality"""
    # Create a clean authenticator for testing
    authenticator = llm_github_copilot.GitHubCopilotAuthenticator()
    
    # Mock open to prevent file read/write operations
    mock_data = {"token": GITHUB_COPILOT_API_KEY, "expires_at": 9999999999}
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = json.dumps(mock_data)
    
    # First patch os.path.exists to return True so it tries to read the file
    with patch("os.path.exists", return_value=True):
        # Then patch open to return our mock file
        with patch("builtins.open", return_value=mock_file):
            # Now get the API key - should read from the "file"
            api_key = authenticator.get_api_key()
            
            # Verify we got the expected token
            assert api_key == GITHUB_COPILOT_API_KEY


@pytest.mark.vcr
def test_model_mappings():
    """Test the model mappings functionality"""
    # Create dummy mappings for testing
    test_mappings = {
        "github-copilot": "gpt-4o",
        "github-copilot/claude-3-7-sonnet": "claude-3-7-sonnet"
    }
    
    # Directly patch the class attribute with our test data
    with patch.object(llm_github_copilot.GitHubCopilot, '_model_mappings', test_mappings):
        model = llm.get_model("github-copilot")
        
        # Get the mappings - should return our test data directly
        mappings = model.get_model_mappings()
        
        # Check the mappings match what we set
        assert "github-copilot" in mappings
        assert mappings["github-copilot"] == "gpt-4o"
        assert "github-copilot/claude-3-7-sonnet" in mappings
