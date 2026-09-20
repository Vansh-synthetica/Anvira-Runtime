"""Test doubles: a fake OpenAI-compatible LLM, a fake Hugging Face, and a fake llama-server."""
from __future__ import annotations

import json
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FAKE_GGUF = b"GGUF" + b"\x00" * 4096   # not a loadable model; the fake llama-server doesn't care


class FakeLLM:
    """OpenAI-compatible server. ``mode``: ok | error | slow. Also serves a fake Hugging Face."""

    def __init__(self):
        self.mode = "ok"
        self.delay = 0.0
        self.requests: list[dict] = []
        self.hf_files = {"acme/tiny-GGUF": {"tiny-Q4_K_M.gguf": FAKE_GGUF},
                         "acme/huge-GGUF": {"huge-Q4_K_M.gguf": FAKE_GGUF}}
        self.hf_sizes = {"huge-Q4_K_M.gguf": 10**15}        # claims a petabyte: exercises the disk-space check
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                p = self.path.split("?")[0]
                if p in ("/health", "/v1/health"):
                    return self._json(200, {"status": "ok"})
                if p == "/v1/models":
                    return self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
                if p == "/api/models":
                    return self._json(200, [{"id": r, "downloads": 10, "likes": 1, "tags": ["gguf"]} for r in outer.hf_files])
                if p.startswith("/api/models/"):
                    repo = p[len("/api/models/"):]
                    files = outer.hf_files.get(repo)
                    if files is None:
                        return self._json(404, {"error": "not found"})
                    return self._json(200, {"id": repo, "siblings": [{"rfilename": n, "size": outer.hf_sizes.get(n, len(b))} for n, b in files.items()]})
                if "/resolve/main/" in p:
                    repo, name = p.strip("/").split("/resolve/main/")
                    body = outer.hf_files.get(repo, {}).get(name)
                    if body is None:
                        return self._json(404, {"error": "no such file"})
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path.endswith("/chat/completions"):
                    outer.requests.append(body)
                    if outer.mode == "error":
                        return self._json(500, {"error": {"message": "fake model exploded"}})
                    if outer.mode == "slow":
                        time.sleep(outer.delay)
                    last = next((m["content"] for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
                    text = f"fake reply to: {last}"
                    if body.get("stream"):
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.end_headers()
                        for piece in ("fake ", "stream ", "reply"):
                            chunk = {"choices": [{"index": 0, "delta": {"content": piece}}]}
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    return self._json(200, {"id": "x", "object": "chat.completion", "model": body.get("model"),
                                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                         "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
                self._json(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self.server.shutdown()


FAKE_LLAMA_SOURCE = '''
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default

port = int(arg("--port"))

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, code, obj):
        d = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        if self.path == "/health": return self._j(200, {"status": "ok"})
        if self.path == "/v1/models": return self._j(200, {"data": [{"id": "local"}]})
        self._j(404, {})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        last = next((m["content"] for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
        self._j(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": "local reply to: " + last},
                                   "finish_reason": "stop"}]})

print("fake llama-server on", port, "model", arg("-m"), flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def make_fake_llama_server(directory: Path) -> Path:
    """Create an executable that behaves like llama-server (a wrapper around a Python script)."""
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "fake_llama.py"
    script.write_text(FAKE_LLAMA_SOURCE, encoding="utf-8")
    if sys.platform == "win32":
        wrapper = directory / "llama-server.cmd"
        wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = directory / "llama-server"
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    return wrapper
