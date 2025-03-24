# llm-github-copilot

A plugin for [LLM](https://llm.datasette.io/) adding support for [GitHub Copilot](https://github.com/features/copilot).

## Installation

You can install this plugin using the LLM command-line tool:

```bash
llm install llm-github-copilot
```

## Authentication

This plugin uses GitHub's device code authentication flow. When you first use the plugin, it will prompt you to visit GitHub and enter a code to authenticate.

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
