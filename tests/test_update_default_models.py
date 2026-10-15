"""Unit tests for scripts/update_default_models.py (no network)."""
import dataclasses
import importlib.util
import json
import pathlib
import socketserver
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("udm", ROOT / "scripts" / "update_default_models.py")
udm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(udm)

HEADER = """#pragma once
constexpr const char *OPENAI_DEFAULT_MODEL = "gpt-6-luna";
constexpr const char *ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5";
constexpr const char *DEEPSEEK_DEFAULT_MODEL = "deepseek-flash";
"""

README = """intro
<!-- BEGIN DEFAULT_MODELS -->
stale
<!-- END DEFAULT_MODELS -->
outro
"""


def by_name(name):
    return next(p for p in udm.PROVIDERS if p.name == name)


class PickLatest(unittest.TestCase):
    def test_openai_takes_highest_version_of_the_cheap_tier(self):
        ids = ["gpt-6-luna", "gpt-5.6-luna", "gpt-6.1-luna", "gpt-6-luna-2026-01-01", "gpt-6-astra", "gpt-6.2-sol"]
        self.assertEqual(udm.pick_latest(ids, by_name("openai")), "gpt-6.1-luna")

    def test_anthropic_ignores_dated_snapshots_and_other_tiers(self):
        ids = ["claude-haiku-4-5", "claude-haiku-4-5-20251001", "claude-haiku-5-5",
               "claude-haiku-5-5-20260601", "claude-sonnet-6-0"]
        self.assertEqual(udm.pick_latest(ids, by_name("anthropic")), "claude-haiku-5-5")

    def test_deepseek_alias_must_still_exist(self):
        self.assertEqual(udm.pick_latest(["deepseek-flash", "deepseek-v4-pro"], by_name("deepseek")), "deepseek-flash")
        with self.assertRaises(udm.NoMatch):
            udm.pick_latest(["deepseek-v4-pro"], by_name("deepseek"))

    def test_no_match_is_an_error_not_a_silent_keep(self):
        with self.assertRaises(udm.NoMatch):
            udm.pick_latest(["gpt-6-astra"], by_name("openai"))


class Rewrite(unittest.TestCase):
    def test_read_defaults(self):
        self.assertEqual(udm.read_defaults(HEADER), {
            "openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5", "deepseek": "deepseek-flash"})

    def test_write_changes_only_what_changed_and_is_idempotent(self):
        new = {"openai": "gpt-6.1-luna", "anthropic": "claude-haiku-5-5", "deepseek": "deepseek-flash"}
        out = udm.write_defaults(HEADER, new)
        self.assertIn('OPENAI_DEFAULT_MODEL = "gpt-6.1-luna"', out)
        self.assertIn('ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5"', out)
        self.assertEqual(udm.write_defaults(out, new), out)

    def test_readme_block_replaced_in_place(self):
        defaults = {"openai": "gpt-6.1-luna", "anthropic": "claude-haiku-5-5", "deepseek": "deepseek-flash"}
        out = udm.update_readme(README, defaults)
        self.assertTrue(out.startswith("intro\n<!-- BEGIN DEFAULT_MODELS -->"))
        self.assertTrue(out.endswith("<!-- END DEFAULT_MODELS -->\noutro\n"))
        self.assertIn("`gpt-6.1-luna`", out)
        self.assertNotIn("stale", out)
        self.assertEqual(udm.update_readme(out, defaults), out)


class Resolve(unittest.TestCase):
    def test_missing_key_skips_provider_and_keeps_current(self):
        current = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5", "deepseek": "deepseek-flash"}
        seen = []

        def fake_fetch(provider, key):
            seen.append(provider.name)
            return ["gpt-6.1-luna"]

        new, notes = udm.resolve(current, {"OPENAI_API_KEY": "k"}, fake_fetch)
        self.assertEqual(seen, ["openai"])
        self.assertEqual(new["openai"], "gpt-6.1-luna")
        self.assertEqual(new["anthropic"], "claude-haiku-5-5")
        self.assertTrue(any("ANTHROPIC_API_KEY" in n for n in notes))

    def test_fetch_failure_propagates(self):
        def boom(provider, key):
            raise udm.ProviderError("401")

        with self.assertRaises(udm.ProviderError):
            udm.resolve({"openai": "x"}, {"OPENAI_API_KEY": "k"}, boom)


class QuickServer(HTTPServer):
    """Skips HTTPServer.server_bind()'s socket.getfqdn(), a slow reverse DNS lookup on CI runners."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


class Fetch(unittest.TestCase):
    """fetch_ids against a local server: auth header style per provider, parsing, error mapping."""

    def serve(self, status, payload):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["headers"] = {k.lower(): v for k, v in self.headers.items()}
                out = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *args):
                pass

        server = QuickServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/v1/models", seen

    def test_bearer_for_openai_style(self):
        url, seen = self.serve(200, {"data": [{"id": "gpt-6-luna"}, {"id": "gpt-6-astra"}, {"nope": 1}]})
        provider = dataclasses.replace(by_name("openai"), models_url=url)
        self.assertEqual(udm.fetch_ids(provider, "sk-test"), ["gpt-6-luna", "gpt-6-astra"])
        self.assertEqual(seen["headers"]["authorization"], "Bearer sk-test")

    def test_x_api_key_for_anthropic(self):
        url, seen = self.serve(200, {"data": [{"id": "claude-haiku-5-5"}]})
        provider = dataclasses.replace(by_name("anthropic"), models_url=url)
        self.assertEqual(udm.fetch_ids(provider, "sk-ant"), ["claude-haiku-5-5"])
        self.assertEqual(seen["headers"]["x-api-key"], "sk-ant")
        self.assertIn("anthropic-version", seen["headers"])
        self.assertNotIn("authorization", seen["headers"])

    def test_http_error_becomes_provider_error_without_leaking_the_key(self):
        url, _ = self.serve(401, {"error": {"message": "bad key sk-test"}})
        provider = dataclasses.replace(by_name("openai"), models_url=url)
        with self.assertRaises(udm.ProviderError) as ctx:
            udm.fetch_ids(provider, "sk-test")
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("sk-test", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
