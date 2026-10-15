![ask banner](./banner.png)

# ASK

[![Release](https://github.com/StevenLi-phoenix/ask/actions/workflows/release.yml/badge.svg)](https://github.com/StevenLi-phoenix/ask/actions/workflows/release.yml)

`ask` is a lightweight CLI that sends chat prompts to OpenAI, Anthropic (Claude Haiku) or DeepSeek. It runs as a single C++17 binary (libcurl + vendored cJSON) and supports both one-shot replies and interactive chat.

- Default model: Claude Haiku (needs `ANTHROPIC_API_KEY`); each provider's default is its cheap tier, see [Providers and default models](#providers-and-default-models)
- Default reasoning effort: `none` (OpenAI only)
- Default temperature: `1.0`
- Default token limit: `128000`

## Build

Dependencies: g++ (C++17), libcurl. cJSON is vendored in `vendor/cjson/`.

macOS:

```bash
brew install curl
make
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y g++ libcurl4-openssl-dev
make
```

This produces the `ask` binary from `ask.cpp`.

## Providers and default models

`ask` talks to three providers through their OpenAI-compatible chat endpoints. Pick one with `-m`: a shortcut selects that provider's default model, any other value is sent as a model id (ids starting with `claude-` go to Anthropic, `deepseek` to DeepSeek, everything else to OpenAI). Without `-m`, `ask` uses Claude Haiku.

<!-- BEGIN DEFAULT_MODELS -->
| Provider | `-m` shortcuts | Default model | API key variable |
|---|---|---|---|
| OpenAI | `openai` | `gpt-6-luna` | `OPENAI_API_KEY` |
| Anthropic | `claude`, `haiku`, `anthropic` | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek`, `flash` | `deepseek-flash` | `DEEPSEEK_API_KEY` |
<!-- END DEFAULT_MODELS -->

```bash
ask "hi"                      # Claude Haiku (the default)
ask -m openai "hi"            # OpenAI's cheap tier
ask -m claude "hi"            # Claude Haiku (also: haiku)
ask -m deepseek "hi"          # DeepSeek Flash (also: flash)
ask -m claude-haiku-4-5 "hi"  # an explicit id keeps its provider
```

The defaults are the cheap, high-volume tier of each provider. They live in `default_models.h` and are bumped automatically, see [Default model updates](#default-model-updates).

## Configure API keys / model

Each provider reads its own key variable (table above) from the environment or from `.env`; the model comes from `-m` or `ASK_GLOBAL_MODEL`:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=...
export ASK_GLOBAL_MODEL=claude  # optional; a shortcut keeps following that provider's default
```

Or write a `.env` (a commented template is in [`.env.example`](.env.example)):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=...
ASK_GLOBAL_MODEL=claude
```

Or persist via flags (updates `.env` and keeps its other lines; the key is stored under the variable of the provider you select):

```bash
./ask --setAPIKey sk-ant-... --setModel haiku
```

`ASK_BASE_URL` (environment or `.env`, the environment wins) replaces the endpoint of whichever provider is selected, e.g. `https://openrouter.ai/api/v1` or a proxy; there is no per-provider override. Model-name validation is skipped for it.

For a gateway such as OpenRouter set `ASK_BASE_URL`, `ASK_MODEL_KEY` (one key for whichever provider is selected; it beats the provider's own variable, `-t` beats it) and use the gateway's model slugs, since the shortcuts above expand to each vendor's own model ids:

```
ASK_BASE_URL=https://openrouter.ai/api/v1
ASK_MODEL_KEY=sk-or-v1-...
ASK_GLOBAL_MODEL=anthropic/claude-haiku-5.5
```

## Usage

One-shot:

```bash
./ask "What is the capital of France?"
```

Interactive:

```bash
./ask -c "Let's chat"
# then type messages; use `exit` to quit, `status` to view token/model info
```

Disable streaming (receive full response at once):

```bash
./ask --no-stream "Tell me a story"
```

Set a custom system prompt:

```bash
./ask -s "You are a pirate" "Hello there"
```

Raw output mode (no spinner, minimal formatting — useful for piping):

```bash
./ask --raw "Give me a JSON example" | jq .
```

Follow-up on the last question without conversation mode:

```bash
./ask "how to list files recursively"
./ask --context last "what about hidden files"
```

Translate a request into a shell command and run it (`-x`). The model is given a `run_shell_command` tool (OpenAI tool calling), so the command comes back as structured arguments rather than text to copy out of a reply. `ask` shows the command and asks before running it:

```bash
$ ./ask -x "iptables show all rules"
# List all iptables rules with counters, numeric output and line numbers.
$ sudo iptables -L -v -n --line-numbers
Run it? [y/N]: y
Chain INPUT (policy ACCEPT 0 packets, 0 bytes)
...
```

- The command runs through `/bin/sh -c`; `ask` exits with the command's exit code.
- `-y` / `--yes` skips the confirmation. The model writes the command, so only use it for requests you would be comfortable running blind.
- `--dry-run` prints just the command to stdout without running it, e.g. `cmd=$(ask -x --dry-run "show listening ports")`. The same happens whenever stdin is not a terminal and `--yes` is not given, since there is nothing to ask on.
- If no shell command fits the request, the model's reply is printed to stderr and nothing runs.
- Not combinable with `-c` or `--context`.

Attach text files inline using `@path`. The file content is injected directly into the prompt text — no vector storage or file uploads to OpenAI. Files over the size limit (default 10KB) trigger an interactive prompt in a terminal, or are silently skipped when piped:

```bash
./ask "Summarize @README.md"
./ask --fileLimit 50000 "Explain @ask.cpp"         # raise limit to 50KB
```

## Flags

| Flag | Description |
|---|---|
| `-h`, `--help` | Display help message |
| `-v`, `--version` | Display version information |
| `-c`, `--continue` | Interactive conversation mode |
| `--context last` | Prepend previous Q&A for lightweight follow-ups |
| `-x`, `--exec` | Translate the request into a shell command (tool call), confirm, then run it |
| `-y`, `--yes` | With `--exec`: run the command without asking first |
| `--dry-run` | With `--exec`: print the command to stdout, do not run it |
| `--no-stream` | Disable SSE streaming (wait for full response) |
| `--raw` | Raw output mode (no spinner, minimal formatting) |
| `-s`, `--system PROMPT` | Set custom system prompt |
| `-t`, `--token TOKEN` | API key for this run |
| `-m`, `--model MODEL` | Model id or shortcut (`openai`, `claude`, `deepseek`; also `haiku`, `flash`) for this run |
| `-T`, `--temperature VAL` | Sampling temperature, 0.0–2.0 (default: 1.0) |
| `-l`, `--tokenLimit NUM` | Max tokens budget (default: 128000) |
| `-F`, `--fileLimit NUM` | `@file` size limit in bytes (default: 10000) |
| `--tokenCount` | Print approximate token count for input and exit |
| `--debug` / `--log LEVEL` | Set log verbosity (`none`, `error`, `warn`, `info`, `debug`) |
| `--logfile FILE` | Log output to a file |
| `--setAPIKey KEY` | Persist API key to `.env` |
| `--setModel MODEL` | Persist model to `.env` |

Token counting is approximate. Streaming prints chunks as they arrive and falls back to full JSON if needed. The CLI shows a "thinking..." spinner and retries once on timeout (60s).

## Default model updates

`.github/workflows/update-default-models.yml` runs every Monday (and on demand). `scripts/update_default_models.py` reads one public documentation page per provider, no API keys or secrets involved, keeps the model ids that match that provider's cheap-tier name pattern (`PROVIDERS` in the script) and takes the highest version. If anything changed it rewrites `default_models.h` and the table above, builds, runs the tests and opens a `chore: bump default models` PR.

Setup, once: let Actions open pull requests.

```bash
gh api -X PUT repos/<owner>/ask/actions/permissions/workflow -F can_approve_pull_request_reviews=true
```

A provider whose page no longer shows a matching model fails the run on purpose: the naming scheme or the page changed and the pattern needs updating. Try it locally with `python3 scripts/update_default_models.py --dry-run`.

## Tests

```bash
make && python3 -I tests/test_update_default_models.py && tests/run.sh
```

`tests/run.sh` drives the binary against a local mock server, so it needs no API keys and touches neither your `.env` nor `~/.cache`.

## Releases

Tags matching `v*` trigger GitHub Actions to build and upload tarballs:

```bash
git tag v1.0
git push origin v1.0
```

Artifacts are published as `ask-<os>-<arch>.tar.gz` with SHA256 checksums for Linux and macOS.

## Contributing

Issues and PRs welcome. Keep changes portable (Linux/macOS) and stay within the C++17 toolchain used in `Makefile`.
