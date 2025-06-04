import os
import pytest
from unittest.mock import patch

# Configure VCR
@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            "Authorization",
            "X-Api-Key",
            "editor-version",
            "editor-plugin-version",
            "Copilot-Integration-Id",
        ],
        "filter_query_parameters": ["key"],
        "decode_compressed_response": True,
    }

# Set up cassette paths for VCR
@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    return os.path.join(os.path.dirname(request.module.__file__), "cassettes")


@pytest.fixture(autouse=True)
def unset_copilot_env_vars():
    """Fixture to automatically unset GitHub Copilot environment variables for tests."""
    env_vars_to_clear = ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]
    with patch.dict(os.environ, {var: "" for var in env_vars_to_clear}, clear=True):
        yield


@pytest.fixture(autouse=True)
def unset_copilot_env_vars():
    """Fixture to automatically unset GitHub Copilot environment variables for tests."""
    env_vars_to_clear = ["GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN"]
    with patch.dict(os.environ, {var: "" for var in env_vars_to_clear}, clear=True):
        yield
