
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

This plugin uses GitHub's device code authentication flow to obtain the initial access_token.   This access_token is what is used to initiate and obtain an api_key that is used for communication with the GitHub Copilot API.  This api_key is automatically obtained and refreshed as needed using the access_token.  The access_token is obtained from the LLM key storage or via the environment variable `GH_COPILOT_TOKEN` or `GITHUB_COPILOT_TOKEN`

Free [plan](https://github.com/features/copilot#pricing) includes up to 2,000 completions and 50 chat requests per month.

### Auth Command Help

```

$ llm github_copilot auth --help
Usage: llm github_copilot auth [OPTIONS] COMMAND [ARGS]...

  Manage GitHub Copilot authentication.

Options:
  --help  Show this message and exit.

Commands:
  login    Authenticate with GitHub Copilot to generate a new access token.
  logout   Remove GitHub Copilot authentication credentials.
  refresh  Force refresh the GitHub Copilot API key.
  status   Check GitHub Copilot authentication status.

```

### Authentication Login

When you run the login command, the plugin will:

1. Start the GitHub device code authentication flow
2. Provide you with a code and URL to visit
3. Wait for you to authenticate on GitHub's website
4. Generate and store an access token
5. Fetch an API key using the access token

Example login output:

```
Usage: llm github_copilot auth login [OPTIONS]

  Authenticate with GitHub Copilot to generate a new access token.

Options:
  -f, --force  Force login even if already authenticated
  --show-only  Perform login but only display tokens, do not save them
  --help       Show this message and exit.
```

```
$ llm github_copilot auth login
Starting GitHub Copilot authentication to generate a new access token...
Please visit https://github.com/login/device and enter code XXXX-XXXX to authenticate GitHub Copilot.

Waiting for authorization... (attempt 1/12)
Authentication successful!
```

You can force a new login and obtain a new access_token even if already authenticated:

```bash
llm github_copilot auth login --force
```

You can also perform the login process and display the obtained access token and API key without saving them to the LLM keystore or the API key file. This is useful if you want to manually manage these tokens or use them in environment variables:
```bash
llm github_copilot auth login --show-only
```
This will output something like:
```
GitHub Copilot: ✓ Authenticated
          User: your_github_username
   AccessToken: Valid
   AccessToken: ghu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       API Key: Valid, expires YYYY-MM-DD HH:MM:SS
       API key: yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

You can set GH_COPILOT_TOKEN or GITHUB_COPILOT_TOKEN to the token value above.
Note: These tokens have NOT been saved to the LLM keystore or API key file.
```

### Authentication Status
```

### Authentication Status

You can check your authentication status with:

```
$ llm github_copilot auth status
GitHub Copilot: ✓ Authenticated
       API Key: Valid, expires 2025-07-15 10:30:00
   AccessToken: Valid, via keystore github_copilot_access_token
```

For more detailed information, use the verbose flag:

```
$ llm github_copilot auth status --verbose
GitHub Copilot: ✓ Authenticated
          User: testuser
   AccessToken: Valid, via keystore github_copilot_access_token
   AccessToken: ghu_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       API Key: Valid, expires 2025-07-15 10:30:00
       API key: abcdef1234567890abcdef1234567890
```

## Usage

Once installed, you can use GitHub Copilot models with the `llm` command:

```bash
# Chat with GitHub Copilot
llm -m github_copilot "Write a Python function that calculates the Fibonacci sequence."

# Specify options like length
llm -m github_copilot "Tell me a joke" -o max_tokens 100
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
export PYTEST_GITHUB_COPILOT_TOKEN=your_token_here
pytest --vcr-record=new_episodes
```
