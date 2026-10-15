// Default model per provider (the cheap, high-volume tier of each).
// Kept current by scripts/update_default_models.py, which the weekly
// .github/workflows/update-default-models.yml runs and turns into a PR.
#pragma once

constexpr const char *OPENAI_DEFAULT_MODEL = "gpt-6-luna";
constexpr const char *ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5";
constexpr const char *DEEPSEEK_DEFAULT_MODEL = "deepseek-flash";
