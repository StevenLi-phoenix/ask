# ASK

[![Release](https://github.com/StevenLi-phoenix/ask/actions/workflows/release.yml/badge.svg)](https://github.com/StevenLi-phoenix/ask/actions/workflows/release.yml)

`ask` is a lightweight CLI that sends chat prompts to OpenAI. It now runs as a single C++17 binary (libcurl + cJSON) and supports both one-shot replies and interactive chat.

- Default model: `gpt-5-nano`
- Default temperature: `1.0`
- Default token limit: `128000`

## Build

Dependencies: g++ (C++17), libcurl, cJSON.

macOS:

```bash
brew install curl cjson
make
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y g++ libcurl4-openssl-dev libcjson-dev
make
```

This produces the `ask` binary. `ask.sh` is a helper wrapper that runs it from the repo root.

## Configure API key / model

You must provide `OPENAI_API_KEY` (and optionally `ASK_GLOBAL_MODEL`):

```bash
export OPENAI_API_KEY=sk-...
export ASK_GLOBAL_MODEL=GPT-5-nano
```

Or write a `.env`:

```
OPENAI_API_KEY=sk-...
ASK_GLOBAL_MODEL=GPT-5-nano
```

Or persist via flags (writes `.env`):

```bash
./ask --setAPIKey sk-... --setModel gpt-4o-mini
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

Attach text files inline using `@path` (up to 10KB, plain text, exact name):

```bash
./ask "Summarize @README.md"
```

## Flags (selected)

- `-c`, `--continue`        interactive conversation
- `--no-stream`             disable SSE streaming
- `-t`, `--token`           API key for this run
- `-m`, `--model`           model for this run (default: GPT-5-nano)
- `-T`, `--temperature`     sampling temperature (default: 1.0)
- `-l`, `--tokenLimit`      max tokens budget (default: 128000, approximate)
- `--tokenCount`            print approximate token count for input and exit
- `--debug` / `--log LEVEL` set log verbosity; `--logfile FILE` to log to file
- `--setAPIKey`, `--setModel` persist to `.env`
- `--help`, `--version`     info

Token counting is approximate; streaming prints chunks as they arrive and falls back to full JSON if needed.
The CLI shows a short “thinking…” spinner and retries once on timeout (60s) so you see progress instead of hanging.

## Releases

Tags matching `v*` trigger GitHub Actions to build and upload tarballs:

```bash
git tag v0.2
git push origin v0.2
```

Artifacts are published as `ask-<os>-<arch>.tar.gz` with SHA256 checksums for Linux and macOS.

## Contributing

Issues and PRs welcome. Keep changes portable (Linux/macOS) and stay within the C++17 toolchain used in `Makefile`.
