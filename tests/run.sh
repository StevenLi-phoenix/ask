#!/usr/bin/env bash
# Black-box tests: provider routing, default models, key lookup and .env handling.
# Runs ./ask against a local mock server; never touches the real ~/.cache or .env.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ASK=${ASK:-$ROOT/ask}
TMP=$(mktemp -d)
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
    rm -rf "$TMP"
}
trap cleanup EXIT

mkdir -p "$TMP/home" "$TMP/work"
LOG=$TMP/requests.jsonl
python3 -I "$ROOT/tests/mock_server.py" "$LOG" "$TMP/port" &
SERVER_PID=$!
for _ in $(seq 50); do [ -s "$TMP/port" ] && break; sleep 0.1; done
PORT=$(cat "$TMP/port")

default_of() { sed -n "s/.*${1}_DEFAULT_MODEL = \"\\(.*\\)\";/\\1/p" "$ROOT/default_models.h"; }
OPENAI_DEFAULT=$(default_of OPENAI)
ANTHROPIC_DEFAULT=$(default_of ANTHROPIC)
DEEPSEEK_DEFAULT=$(default_of DEEPSEEK)

# run [ENV=VAL ...] -- ask-args...   (clean env, cwd = $TMP/work, endpoint = mock; exit code -> $RC)
run() {
    : > "$LOG"
    RC=0
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done
    shift
    (cd "$TMP/work" && env -i PATH="$PATH" HOME="$TMP/home" ASK_BASE_URL="http://127.0.0.1:$PORT" \
        ${envs[@]+"${envs[@]}"} "$ASK" --no-stream --raw "$@") </dev/null 2>"$TMP/stderr" || RC=$?
}
# field NAME  -> value from the last logged request ("" if absent)
field() {
    python3 -I - "$LOG" "$1" <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
req = json.loads(lines[-1]) if lines else {}
body = req.get("body", {})
name = sys.argv[2]
print({"path": req.get("path"), "auth": req.get("authorization"),
       "model": body.get("model"), "effort": body.get("reasoning_effort"),
       "tools": "yes" if "tools" in body else ""}[name] or "")
PY
}

FAILS=0
expect() { # desc actual expected
    if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: got '$2', want '$3'"; FAILS=$((FAILS + 1)); fi
}

# 1. default = Claude Haiku: Anthropic key, /chat/completions, no reasoning_effort
run ANTHROPIC_API_KEY=sk-anth -- hi >/dev/null
expect "default model" "$(field model)" "$ANTHROPIC_DEFAULT"
expect "default path" "$(field path)" "/chat/completions"
expect "default key" "$(field auth)" "Bearer sk-anth"
expect "default omits reasoning_effort" "$(field effort)" ""

# 1b. OpenAI is one flag away and is the only provider that gets reasoning_effort
run OPENAI_API_KEY=sk-openai -- -m openai hi >/dev/null
expect "openai model" "$(field model)" "$OPENAI_DEFAULT"
expect "openai key" "$(field auth)" "Bearer sk-openai"
expect "openai sends reasoning_effort" "$(field effort)" "none"

# 2. haiku alias -> Anthropic default, own key, no reasoning_effort
run OPENAI_API_KEY=sk-openai ANTHROPIC_API_KEY=sk-anth -- -m haiku hi >/dev/null
expect "haiku model" "$(field model)" "$ANTHROPIC_DEFAULT"
expect "haiku key" "$(field auth)" "Bearer sk-anth"
expect "anthropic omits reasoning_effort" "$(field effort)" ""

# 3. flash alias -> DeepSeek default
run DEEPSEEK_API_KEY=sk-ds -- -m flash hi >/dev/null
expect "flash model" "$(field model)" "$DEEPSEEK_DEFAULT"
expect "flash key" "$(field auth)" "Bearer sk-ds"
expect "deepseek omits reasoning_effort" "$(field effort)" ""

# 3b. bare provider names select that provider's default (claude = Anthropic, anthropic accepted too)
run ANTHROPIC_API_KEY=sk-anth -- -m claude hi >/dev/null
expect "-m claude model" "$(field model)" "$ANTHROPIC_DEFAULT"
expect "-m claude key" "$(field auth)" "Bearer sk-anth"
run ANTHROPIC_API_KEY=sk-anth -- -m anthropic hi >/dev/null
expect "-m anthropic model" "$(field model)" "$ANTHROPIC_DEFAULT"
run DEEPSEEK_API_KEY=sk-ds -- -m deepseek hi >/dev/null
expect "-m deepseek model" "$(field model)" "$DEEPSEEK_DEFAULT"
expect "-m deepseek key" "$(field auth)" "Bearer sk-ds"

# 4. provider inferred from the model id prefix; explicit ids pass through untouched
run ANTHROPIC_API_KEY=sk-anth -- -m claude-haiku-4-5 hi >/dev/null
expect "claude-* model kept" "$(field model)" "claude-haiku-4-5"
expect "claude-* uses anthropic key" "$(field auth)" "Bearer sk-anth"
run DEEPSEEK_API_KEY=sk-ds -- -m deepseek-reasoner hi >/dev/null
expect "deepseek-* uses deepseek key" "$(field auth)" "Bearer sk-ds"
run OPENAI_API_KEY=sk-openai -- -m gpt-5.4-mini hi >/dev/null
expect "other ids stay on openai" "$(field auth)" "Bearer sk-openai"

# 5. persistent selection through ASK_GLOBAL_MODEL
run ANTHROPIC_API_KEY=sk-anth ASK_GLOBAL_MODEL=haiku -- hi >/dev/null
expect "ASK_GLOBAL_MODEL=haiku" "$(field model)" "$ANTHROPIC_DEFAULT"

# 6. missing key for the selected provider names the right variable
run OPENAI_API_KEY=sk-openai -- hi >/dev/null
expect "default needs ANTHROPIC_API_KEY" "$(grep -c ANTHROPIC_API_KEY "$TMP/stderr")" "1"
run OPENAI_API_KEY=sk-openai -- -m haiku hi >/dev/null
expect "missing key exits non-zero" "$([ "$RC" -ne 0 ] && echo yes || echo no)" "yes"
expect "missing key names ANTHROPIC_API_KEY" "$(grep -c ANTHROPIC_API_KEY "$TMP/stderr")" "1"

# 7. keys come from .env too, and --set* only touches what it was asked to
printf '# my note\nOPENAI_API_KEY=sk-env-openai\nANTHROPIC_API_KEY=sk-env-anth\n' > "$TMP/work/.env"
run -- -m haiku hi >/dev/null
expect ".env supplies provider key" "$(field auth)" "Bearer sk-env-anth"
run -- --setModel flash >/dev/null
expect "--setModel keeps comment" "$(grep -c '^# my note$' "$TMP/work/.env")" "1"
expect "--setModel keeps other keys" "$(grep -c '^ANTHROPIC_API_KEY=sk-env-anth$' "$TMP/work/.env")" "1"
expect "--setModel writes model" "$(grep -c '^ASK_GLOBAL_MODEL=flash$' "$TMP/work/.env")" "1"
run -- --setAPIKey sk-env-ds --setModel flash >/dev/null
expect "--setAPIKey stored under active provider" "$(grep -c '^DEEPSEEK_API_KEY=sk-env-ds$' "$TMP/work/.env")" "1"
expect "--setAPIKey leaves openai key" "$(grep -c '^OPENAI_API_KEY=sk-env-openai$' "$TMP/work/.env")" "1"

# 7b. ASK_BASE_URL is read from .env too; the environment wins over .env
printf 'ASK_BASE_URL=http://127.0.0.1:%s\nANTHROPIC_API_KEY=sk-env-anth\n' "$PORT" > "$TMP/work/.env"
run ASK_BASE_URL= -- hi >/dev/null
expect "ASK_BASE_URL from .env reaches the endpoint" "$(field path)" "/chat/completions"
expect "ASK_BASE_URL from .env keeps provider key" "$(field auth)" "Bearer sk-env-anth"
printf 'ASK_BASE_URL=http://127.0.0.1:1\nANTHROPIC_API_KEY=sk-env-anth\n' > "$TMP/work/.env"
run -- hi >/dev/null
expect "environment ASK_BASE_URL beats .env" "$(field path)" "/chat/completions"

# 7c. ASK_MODEL_KEY: one key for whichever provider / gateway is selected.
#     It beats the provider's own variable and loses only to -t.
run ASK_MODEL_KEY=sk-global -- hi >/dev/null
expect "ASK_MODEL_KEY serves the default provider" "$(field auth)" "Bearer sk-global"
run ASK_MODEL_KEY=sk-global DEEPSEEK_API_KEY=sk-ds -- -m flash hi >/dev/null
expect "ASK_MODEL_KEY beats the provider variable" "$(field auth)" "Bearer sk-global"
printf 'ASK_MODEL_KEY=sk-env-global\n' > "$TMP/work/.env"
run -- -m flash hi >/dev/null
expect "ASK_MODEL_KEY from .env" "$(field auth)" "Bearer sk-env-global"
run ASK_MODEL_KEY=sk-global -- -t sk-cli hi >/dev/null
expect "-t beats ASK_MODEL_KEY" "$(field auth)" "Bearer sk-cli"

# 7d. .env.example is a safe starting point: copied as-is it changes nothing, empty keys count as unset
cp "$ROOT/.env.example" "$TMP/work/.env"
run ANTHROPIC_API_KEY=sk-anth -- hi >/dev/null
expect ".env.example keeps the default model" "$(field model)" "$ANTHROPIC_DEFAULT"
expect ".env.example leaves the env key in charge" "$(field auth)" "Bearer sk-anth"
run -- hi >/dev/null
expect ".env.example empty key is not sent" "$(grep -c ANTHROPIC_API_KEY "$TMP/stderr")" "1"

# 8. `-v` must not pretend a key exists when none is configured
rm -f "$TMP/work/.env"
out=$(run -- -v)
expect "-v without key says so" "$(printf '%s\n' "$out" | grep -c '^API Key: (not set)$')" "1"
out=$(run ANTHROPIC_API_KEY=sk-test-1234567890 -- -v)
expect "-v masks a present key" "$(printf '%s\n' "$out" | grep -c '^API Key: sk-te\.\.\.7890$')" "1"
out=$(run ANTHROPIC_API_KEY=abc12345 -- -v)
expect "-v short key shows length" "$(printf '%s\n' "$out" | grep -c '^API Key: \*\*\*\* (length:8)$')" "1"
out=$(run ANTHROPIC_API_KEY=abcdefgh1234 -- -v)
expect "-v 12-char key still shows length" "$(printf '%s\n' "$out" | grep -c '^API Key: \*\*\*\* (length:12)$')" "1"
out=$(run ANTHROPIC_API_KEY=abcdefgh12345 -- -v)
expect "-v 13-char key is masked head/tail" "$(printf '%s\n' "$out" | grep -c '^API Key: abcde\.\.\.2345$')" "1"

# 9. -x / --exec goes through the same provider, key and endpoint logic
rm -f "$TMP/work/.env"
out=$(run ANTHROPIC_API_KEY=sk-anth -- -x --dry-run "say hello")
expect "-x --dry-run prints the tool-call command" "$out" "echo hello"
expect "-x sends the tool" "$(field tools)" "yes"
expect "-x uses the default provider's model" "$(field model)" "$ANTHROPIC_DEFAULT"
expect "-x uses the default provider's key" "$(field auth)" "Bearer sk-anth"
expect "-x omits reasoning_effort for anthropic" "$(field effort)" ""
out=$(run OPENAI_API_KEY=sk-openai -- -m openai -x --dry-run "say hello")
expect "-x on openai sends reasoning_effort none (needed for tools)" "$(field effort)" "none"

[ "$FAILS" -eq 0 ] && echo "all passed" || { echo "$FAILS failed"; exit 1; }
