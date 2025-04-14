
[![PyPI](https://img.shields.io/pypi/v/llm-github-copilot.svg)](https://pypi.org/project/llm-github-copilot/)
[![Changelog](https://img.shields.io/github/v/release/jmdaly/llm-github-copilot?include_prereleases&label=changelog)](https://github.com/jmdaly/llm-github-copilot/releases)
[![Tests](https://github.com/jmdaly/llm-github-copilot/actions/workflows/test.yml/badge.svg)](https://github.com/jmdaly/llm-github-copilot/actions/workflows/test.yml)


# llm-github-copilot

A plugin for [LLM](https://llm.datasette.io/) adding support for [GitHub Copilot](https://github.com/features/copilot).

## Installation

You can install this plugin using the LLM command-line tool:

```bash
llm install llm-github-copilot
```

## Authentication

This plugin uses GitHub's device code authentication flow. When you first use the plugin, it will prompt you to visit GitHub and enter a code to authenticate.

You can also manage authentication using the CLI:

```bash
# Login to GitHub Copilot
llm github-copilot auth login

# Check authentication status
llm github-copilot auth status

# Force refresh the API key
llm github-copilot auth refresh

# Logout and remove credentials
llm github-copilot auth logout
```

### Auth Command Help

```
$ llm github-copilot auth --help
Usage: llm github-copilot auth [OPTIONS] COMMAND [ARGS]...

  Manage GitHub Copilot authentication.

Options:
  --help  Show this message and exit.

Commands:
  login    Authenticate with GitHub Copilot to generate a new access token.
  logout   Remove GitHub Copilot authentication credentials.
  refresh  Force refresh the GitHub Copilot API key.
  status   Check GitHub Copilot authentication status.
```

### Status Command Output

```
$ llm github-copilot auth status
GitHub Copilot authentication: ✓ Authenticated

$ llm github-copilot auth status --verbose
GitHub Copilot authentication: ✓ Authenticated
Access token: ghu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (from LLM key storage)
API key expires: 2025-04-14 12:34:56
API key: gcp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Free [plan](https://github.com/features/copilot#pricing) includes up to 2,000 completions and 50 chat requests per month.


## Usage

Once installed, you can use GitHub Copilot models with the `llm` command:

```bash
# Chat with GitHub Copilot
llm -m github-copilot "Write a Python function that calculates the Fibonacci sequence."

# Specify options like length
llm -m github-copilot "Tell me a joke" -o max_tokens 100
```

## Options

The GitHub Copilot plugin supports the following options:

- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `temperature`: Controls randomness in the output (default: 0.7)

## Development

To develop this plugin:

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-github-copilot.git
cd llm-github-copilot

# Install in development mode
llm install -e .
```

## Testing

To run the tests:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest
```

If you want to record new VCR cassettes for tests, set your API key:

```bash
export PYTEST_GITHUB_COPILOT_API_KEY=your_api_key_here
pytest --vcr-record=new_episodes
```
