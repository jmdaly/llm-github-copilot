import os
import pytest

# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest-asyncio to use function scope for async fixtures."""
    config.option.asyncio_default_fixture_loop_scope = "function"

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
