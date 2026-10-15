#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock used by tests/run.sh.

Usage: mock_server.py REQUEST_LOG PORT_FILE
Every POST is appended to REQUEST_LOG as one JSON line; GETs (e.g. /models) 404.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

request_log, port_file = sys.argv[1], sys.argv[2]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with open(request_log, "a") as f:
            f.write(json.dumps({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(raw),
            }) + "\n")
        message = {"role": "assistant", "content": "pong"}
        if "tools" in json.loads(raw):  # -x / --exec asks for a run_shell_command tool call
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {
                    "name": "run_shell_command",
                    "arguments": json.dumps({"command": "echo hello", "explanation": "say hello"})}}]}
        out = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
with open(port_file, "w") as f:
    f.write(str(server.server_port))
server.serve_forever()
