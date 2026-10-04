![ask banner](./banner.png)

# ASK

[![Release](https://github.com/StevenLi-phoenix/ask/actions/workflows/release.yml/badge.svg)](https://github.com/StevenLi-phoenix/ask/actions/workflows/release.yml)

`ask` is a lightweight CLI that sends chat prompts to OpenAI. It runs as a single C++17 binary (libcurl + vendored cJSON) and supports both one-shot replies and interactive chat.

- Default model: `gpt-5.5`
- Default reasoning effort: `none`
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

## Configure API key / model

You must provide `OPENAI_API_KEY` (and optionally `ASK_GLOBAL_MODEL`):

```bash
export OPENAI_API_KEY=sk-...
export ASK_GLOBAL_MODEL=gpt-5.5
```

Or write a `.env`:

```
OPENAI_API_KEY=sk-...
ASK_GLOBAL_MODEL=gpt-5.5
```

Or persist via flags (writes `.env`):

```bash
./ask --setAPIKey sk-... --setModel gpt-5.5
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
| `-m`, `--model MODEL` | Model for this run (default: `gpt-5.5`) |
| `-T`, `--temperature VAL` | Sampling temperature, 0.0–2.0 (default: 1.0) |
| `-l`, `--tokenLimit NUM` | Max tokens budget (default: 128000) |
| `-F`, `--fileLimit NUM` | `@file` size limit in bytes (default: 10000) |
| `--tokenCount` | Print approximate token count for input and exit |
| `--debug` / `--log LEVEL` | Set log verbosity (`none`, `error`, `warn`, `info`, `debug`) |
| `--logfile FILE` | Log output to a file |
| `--setAPIKey KEY` | Persist API key to `.env` |
| `--setModel MODEL` | Persist model to `.env` |

Token counting is approximate. Streaming prints chunks as they arrive and falls back to full JSON if needed. The CLI shows a "thinking..." spinner and retries once on timeout (60s).

## Releases

Tags matching `v*` trigger GitHub Actions to build and upload tarballs:

```bash
git tag v1.0
git push origin v1.0
```

Artifacts are published as `ask-<os>-<arch>.tar.gz` with SHA256 checksums for Linux and macOS.

## Contributing

Issues and PRs welcome. Keep changes portable (Linux/macOS) and stay within the C++17 toolchain used in `Makefile`.
