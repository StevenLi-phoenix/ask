"""Unit tests for scripts/update_default_models.py (no network)."""
import dataclasses
import importlib.util
import pathlib
import socketserver
import threading
import unittest
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

CURRENT = {"openai": "gpt-6-luna", "anthropic": "claude-haiku-5-5", "deepseek": "deepseek-flash"}


def by_name(name):
    return next(p for p in udm.PROVIDERS if p.name == name)


class Extract(unittest.TestCase):
    def test_model_ids_come_out_of_markdown_html_and_prose(self):
        text = ('Use [GPT-6 Luna](/api/docs/models/gpt-6-luna) or `gpt-6.1-luna`. Older: '
                'claude-haiku-4-5-20251001, "deepseek-flash"&quot; and claude-haiku-5-5.')
        ids = udm.extract_ids(text)
        for want in ("gpt-6-luna", "gpt-6.1-luna", "claude-haiku-4-5-20251001", "deepseek-flash", "claude-haiku-5-5"):
            self.assertIn(want, ids)


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
        self.assertEqual(udm.read_defaults(HEADER), CURRENT)

    def test_write_changes_only_what_changed_and_is_idempotent(self):
        new = dict(CURRENT, openai="gpt-6.1-luna")
        out = udm.write_defaults(HEADER, new)
        self.assertIn('OPENAI_DEFAULT_MODEL = "gpt-6.1-luna"', out)
        self.assertIn('ANTHROPIC_DEFAULT_MODEL = "claude-haiku-5-5"', out)
        self.assertEqual(udm.write_defaults(out, new), out)

    def test_readme_block_replaced_in_place(self):
        out = udm.update_readme(README, dict(CURRENT, openai="gpt-6.1-luna"))
        self.assertTrue(out.startswith("intro\n<!-- BEGIN DEFAULT_MODELS -->"))
        self.assertTrue(out.endswith("<!-- END DEFAULT_MODELS -->\noutro\n"))
        self.assertIn("`gpt-6.1-luna`", out)
        self.assertNotIn("stale", out)
        self.assertEqual(udm.update_readme(out, dict(CURRENT, openai="gpt-6.1-luna")), out)


class Resolve(unittest.TestCase):
    PAGES = {"openai": "[Luna](/m/gpt-6-luna) [Luna 6.1](/m/gpt-6.1-luna) gpt-6-astra",
             "anthropic": "claude-haiku-5-5 claude-haiku-4-5-20251001",
             "deepseek": "model deepseek-flash deepseek-v4-pro"}

    def fetch(self, provider):
        return udm.extract_ids(self.PAGES[provider.name])

    def test_every_provider_is_resolved_from_its_page(self):
        new, notes = udm.resolve(CURRENT, self.fetch)
        self.assertEqual(new, dict(CURRENT, openai="gpt-6.1-luna"))
        self.assertTrue(any("gpt-6-luna -> gpt-6.1-luna" in n for n in notes))
        self.assertTrue(any("claude-haiku-5-5 (unchanged)" in n for n in notes))

    def test_page_without_a_match_is_an_error(self):
        self.PAGES = dict(self.PAGES, anthropic="nothing useful here")
        with self.assertRaises(udm.NoMatch):
            udm.resolve(CURRENT, self.fetch)

    def test_fetch_failure_propagates(self):
        def boom(provider):
            raise udm.ProviderError("403")

        with self.assertRaises(udm.ProviderError):
            udm.resolve(CURRENT, boom)


class QuickServer(HTTPServer):
    """Skips HTTPServer.server_bind()'s socket.getfqdn(), a slow reverse DNS lookup on CI runners."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


class Fetch(unittest.TestCase):
    """fetch_ids against a local server: plain GET, no credentials, error mapping."""

    def serve(self, status, body):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["headers"] = {k.lower(): v for k, v in self.headers.items()}
                out = body.encode()
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
        return f"http://127.0.0.1:{server.server_port}/models.md", seen

    def test_reads_ids_from_the_public_page_without_credentials(self):
        url, seen = self.serve(200, "# Models\n\nModel ID: `gpt-6-luna`, also gpt-6-astra.\n")
        provider = dataclasses.replace(by_name("openai"), source_url=url)
        ids = udm.fetch_ids(provider)
        self.assertIn("gpt-6-luna", ids)
        self.assertIn("gpt-6-astra", ids)
        self.assertNotIn("authorization", seen["headers"])
        self.assertNotIn("x-api-key", seen["headers"])
        self.assertIn("user-agent", seen["headers"])

    def test_http_error_becomes_provider_error(self):
        url, _ = self.serve(403, "blocked")
        provider = dataclasses.replace(by_name("openai"), source_url=url)
        with self.assertRaises(udm.ProviderError) as ctx:
            udm.fetch_ids(provider)
        self.assertIn("403", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
