#!/usr/bin/env python3
"""Bump the default models in default_models.h and the README table.

For each provider the script lists the models its API exposes, keeps the ones that
match the provider's "cheap tier" naming pattern (PROVIDERS below) and picks the
highest version. A provider whose API key is not in the environment is skipped; a
provider with no matching model is an error, because it means the naming scheme
changed and the pattern needs a human.

Stdlib only; deterministic; no LLM involved.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
README_BEGIN = "<!-- BEGIN DEFAULT_MODELS -->"
README_END = "<!-- END DEFAULT_MODELS -->"


class NoMatch(Exception):
    pass


class ProviderError(Exception):
    pass


@dataclass(frozen=True)
class Provider:
    name: str
    shortcuts: tuple
    const: str
    key_env: str
    models_url: str
    pattern: re.Pattern
    anthropic_auth: bool = False


# Patterns pick the cheap tier. When a provider renames its tiers, edit the pattern.
PROVIDERS = (
    Provider("openai", ("openai",), "OPENAI_DEFAULT_MODEL", "OPENAI_API_KEY",
             "https://api.openai.com/v1/models",
             re.compile(r"^gpt-(\d+(?:\.\d+)*)-luna$")),
    # {1,2} digits per component keeps dated snapshots (claude-haiku-4-5-20251001) out.
    Provider("anthropic", ("claude", "haiku", "anthropic"), "ANTHROPIC_DEFAULT_MODEL", "ANTHROPIC_API_KEY",
             "https://api.anthropic.com/v1/models?limit=1000",
             re.compile(r"^claude-haiku-(\d{1,2}(?:-\d{1,2})?)$"), anthropic_auth=True),
    # DeepSeek ships a stable alias that always points at the latest Flash.
    Provider("deepseek", ("deepseek", "flash"), "DEEPSEEK_DEFAULT_MODEL", "DEEPSEEK_API_KEY",
             "https://api.deepseek.com/models",
             re.compile(r"^deepseek-flash$")),
)

LABELS = {"openai": "OpenAI", "anthropic": "Anthropic", "deepseek": "DeepSeek"}


def version_key(model_id, provider):
    match = provider.pattern.match(model_id)
    if match is None or provider.pattern.groups == 0:
        return ()
    return tuple(int(part) for part in re.split(r"[.-]", match.group(1)))


def pick_latest(ids, provider):
    candidates = [i for i in ids if provider.pattern.match(i)]
    if not candidates:
        raise NoMatch(f"{provider.name}: no listed model matches {provider.pattern.pattern}")
    return max(candidates, key=lambda i: version_key(i, provider))


def read_defaults(header_text):
    out = {}
    for p in PROVIDERS:
        m = re.search(rf'{p.const}\s*=\s*"([^"]+)"', header_text)
        if m is None:
            raise ValueError(f"{p.const} not found in header")
        out[p.name] = m.group(1)
    return out


def write_defaults(header_text, new):
    for p in PROVIDERS:
        header_text = re.sub(rf'({p.const}\s*=\s*")[^"]+(")', rf'\g<1>{new[p.name]}\g<2>', header_text)
    return header_text


def render_readme_block(defaults):
    rows = ["| Provider | `-m` shortcuts | Default model | API key variable |", "|---|---|---|---|"]
    for p in PROVIDERS:
        shortcuts = ", ".join(f"`{s}`" for s in p.shortcuts)
        rows.append(f"| {LABELS[p.name]} | {shortcuts} | `{defaults[p.name]}` | `{p.key_env}` |")
    return "\n".join(rows)


def update_readme(text, defaults):
    start, end = text.find(README_BEGIN), text.find(README_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError("README markers not found")
    return text[:start + len(README_BEGIN)] + "\n" + render_readme_block(defaults) + "\n" + text[end:]


def fetch_ids(provider, key):
    headers = {"User-Agent": "ask-default-models-updater"}
    if provider.anthropic_auth:
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(provider.models_url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        e.close()
        raise ProviderError(f"{provider.name}: HTTP {e.code} from {provider.models_url}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ProviderError(f"{provider.name}: {type(e).__name__} fetching models") from None
    return [m["id"] for m in payload.get("data", []) if isinstance(m.get("id"), str)]


def resolve(current, env, fetch):
    """Return (new_defaults, notes). Providers without a key keep their current default."""
    new, notes = dict(current), []
    for p in PROVIDERS:
        key = env.get(p.key_env, "")
        if not key:
            notes.append(f"{p.name}: {p.key_env} not set, keeping {current.get(p.name)}")
            continue
        latest = pick_latest(fetch(p, key), p)
        new[p.name] = latest
        notes.append(f"{p.name}: {current.get(p.name)} -> {latest}" if latest != current.get(p.name)
                     else f"{p.name}: {latest} (unchanged)")
    return new, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--header", type=pathlib.Path, default=ROOT / "default_models.h")
    ap.add_argument("--readme", type=pathlib.Path, default=ROOT / "README.md")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args(argv)

    header = args.header.read_text()
    current = read_defaults(header)
    if not any(os.environ.get(p.key_env) for p in PROVIDERS):
        print("::error::no provider API key in the environment, nothing was checked", file=sys.stderr)
        return 1
    try:
        new, notes = resolve(current, os.environ, fetch_ids)
    except (NoMatch, ProviderError) as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1

    for note in notes:
        print(note)
    changed = new != current
    if changed and not args.dry_run:
        args.header.write_text(write_defaults(header, new))
        args.readme.write_text(update_readme(args.readme.read_text(), new))
    if out := os.environ.get("GITHUB_OUTPUT"):
        with open(out, "a") as f:
            f.write(f"changed={'true' if changed else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
