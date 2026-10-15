# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Anthropic (`claude-haiku-5-5`) and DeepSeek (`deepseek-flash`) next to OpenAI. `-m openai` / `-m claude` / `-m deepseek` (also `haiku`, `flash`, `anthropic`) pick a provider's default; `claude-*` and `deepseek*` ids are routed to their provider automatically. Each provider reads its own key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`) from the environment or `.env`.
- `ASK_BASE_URL` (environment or `.env`) to point the selected provider at any OpenAI-compatible endpoint.
- `ASK_MODEL_KEY` (environment or `.env`): one API key for whichever provider or gateway is selected; overrides the provider's own key variable, `-t` overrides it.
- Weekly workflow `update-default-models.yml` that resolves the newest cheap-tier model per provider (`scripts/update_default_models.py`) and opens a PR bumping `default_models.h` and the README table.
- `.env.example` documenting the provider keys, `ASK_GLOBAL_MODEL` and the gateway variables.
- `ci.yml` and `tests/` (mock-server black-box tests for routing, keys and `.env`; unit tests for the updater).

### Changed
- Default model is now Claude Haiku (`claude-haiku-5-5`, needs `ANTHROPIC_API_KEY`); it was OpenAI `gpt-5.5`. OpenAI's own default is now the cheap tier `gpt-6-luna` (`-m openai`). A model pinned through `ASK_GLOBAL_MODEL` (env or `.env`) still wins.
- Default models live in `default_models.h` instead of a constant in `ask.cpp`.
- `reasoning_effort` is only sent to OpenAI; model-name validation only runs against the official OpenAI endpoint.
- `~` paths resolve through `$HOME` before the passwd entry.

### Fixed
- `ask -v` / `--debug` printed `API Key: ****` even when no key was configured; it now says `(not set)`, and a key of 8 characters or fewer shows `**** (length:N)`.
- `--setModel` / `--setAPIKey` rewrote `.env` with only two lines, dropping every other entry and comment; they now update just the requested keys. `--setModel` also no longer copies a key from the environment into `.env`.
