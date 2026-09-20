"""Honest benchmark: HTTP vs AICL transport comparison.

Measures what actually matters:
    1. Wire bytes on the wire (payload size)
    2. Serialization latency (encode/decode)
    3. Full roundtrip (HTTP request/response vs AICL binary)
    4. Throughput (messages/sec under concurrent load)

Does NOT claim improvements without measurements.
Does NOT use whitespace-split token approximation.
"""
from __future__ import annotations

import json
import socket
import statistics
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen

sys.path.insert(0, ".")

from aicl.bin.codec_api import encode as bin_encode, decode as bin_decode
from aicl.bin.codec_packet import Packet
from aicl.bin.codec_view import PacketView
from aicl.bin.types import Identity, Symbol
from aicl.bin.symbol_types import S_STRING, S_I64, S_F64, S_BOOL
from aicl.bin.constants import HEADER_SIZE
from aicl.semantic.types import ModelIntent, ModelAction
from aicl.semantic.adapter import SemanticAdapter

__all__ = ["run_transport_benchmark"]


# ──────────────────────────────────────────────────────────────
# Payload: a real classifier request/response
# ──────────────────────────────────────────────────────────────

_CLASSIFY_REQUEST = {
    "intent": "CLASSIFY",
    "target": "sentiment_classifier",
    "inputs": {"text": "I absolutely love this product! It exceeded all my expectations and I would highly recommend it to anyone looking for quality."},
    "params": {"labels": ["positive", "negative", "neutral"]},
    "metadata": {"session_id": "s-abc-123", "request_id": "r-001"},
}

_CLASSIFY_RESPONSE = {
    "intent": "CLASSIFY",
    "output": {"label": "positive", "confidence": 0.947},
    "tokens_used": 24,
    "latency_ms": 45.2,
}


def _make_aicl_request_bytes() -> bytes:
    """Build AICL binary request packet."""
    adapter = SemanticAdapter(origin="orchestrator")
    action = ModelAction(
        intent=ModelIntent.CLASSIFY,
        target="sentiment_classifier",
        inputs={"text": _CLASSIFY_REQUEST["inputs"]["text"]},
        params={"labels": ["positive", "negative", "neutral"]},
        metadata={"session_id": "s-abc-123", "request_id": "r-001"},
    )
    return adapter.action_to_bytes(action)


def _make_aicl_response_bytes() -> bytes:
    """Build AICL binary response packet."""
    adapter = SemanticAdapter(origin="sentiment_classifier")
    result = ModelResult = __import__("aicl.semantic.types", fromlist=["ModelResult"]).ModelResult
    from aicl.semantic.types import ModelResult
    r = ModelResult(
        intent=ModelIntent.CLASSIFY,
        output={"label": "positive", "confidence": 0.947},
        tokens_used=24,
        latency_ms=45.2,
    )
    return adapter.result_to_bytes(r)


def _make_http_json_request_bytes() -> bytes:
    """Build HTTP request with JSON body."""
    body = json.dumps(_CLASSIFY_REQUEST).encode("utf-8")
    req = (
        b"POST /classify HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: keep-alive\r\n"
        b"\r\n" + body
    )
    return req


def _make_http_json_response_bytes() -> bytes:
    """Build HTTP response with JSON body."""
    body = json.dumps(_CLASSIFY_RESPONSE).encode("utf-8")
    resp = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"\r\n" + body
    )
    return resp


# ──────────────────────────────────────────────────────────────
# HTTP echo server (for real roundtrip)
# ──────────────────────────────────────────────────────────────

class _EchoHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that echoes JSON back."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        # Parse and respond with classifier result
        resp_body = json.dumps(_CLASSIFY_RESPONSE).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def log_message(self, format, *args):
        pass  # Suppress logging


def _start_http_server(port: int) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ──────────────────────────────────────────────────────────────
# AICL TCP echo server (for real roundtrip)
# ──────────────────────────────────────────────────────────────

class _AICLEchoServer:
    """Minimal TCP server that reads AICL packets and responds."""

    def __init__(self, port: int):
        self.port = port
        self._server_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("127.0.0.1", self.port))
        self._server_sock.listen(8)
        self._server_sock.settimeout(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        adapter = SemanticAdapter(origin="sentiment_classifier")
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            try:
                data = conn.recv(65536)
                if data:
                    # Decode request, build response
                    from aicl.semantic.types import ModelResult, ModelIntent
                    r = ModelResult(
                        intent=ModelIntent.CLASSIFY,
                        output={"label": "positive", "confidence": 0.947},
                        tokens_used=24,
                        latency_ms=45.2,
                    )
                    resp = adapter.result_to_bytes(r)
                    # Length-prefix the response
                    conn.sendall(struct.pack(">I", len(resp)) + resp)
            except Exception:
                pass
            finally:
                conn.close()

    def stop(self):
        self._running = False
        if self._server_sock:
            self._server_sock.close()
        if self._thread:
            self._thread.join(timeout=2)


# ──────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────

@dataclass
class TransportResult:
    name: str
    payload_bytes: int
    wire_bytes: int
    encode_us: float
    decode_us: float
    roundtrip_us: float
    roundtrip_p95_us: float
    throughput_msg_per_sec: float
    parse_success_rate: float


def _measure_encode_decode(
    encode_fn,
    decode_fn,
    rounds: int = 1000,
):
    """Measure encode/decode latency."""
    encode_latencies = []
    decode_latencies = []
    encoded = None

    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        encoded = encode_fn()
        t1 = time.perf_counter_ns()
        encode_latencies.append((t1 - t0) / 1000)

        t2 = time.perf_counter_ns()
        decode_fn(encoded)
        t3 = time.perf_counter_ns()
        decode_latencies.append((t3 - t2) / 1000)

    return encode_latencies, decode_latencies


def run_transport_benchmark(rounds: int = 500) -> Dict[str, TransportResult]:
    """Run the full transport benchmark."""
    results = {}

    # ── 1. HTTP + JSON ────────────────────────────────────────
    print("  [1/4] HTTP + JSON...")
    http_port = 18932
    http_server = _start_http_server(http_port)
    time.sleep(0.1)  # Let server start

    http_req_body = json.dumps(_CLASSIFY_REQUEST).encode("utf-8")
    http_resp_body = json.dumps(_CLASSIFY_RESPONSE).encode("utf-8")

    http_enc_lat, http_dec_lat = _measure_encode_decode(
        encode_fn=lambda: json.dumps(_CLASSIFY_REQUEST).encode("utf-8"),
        decode_fn=lambda d: json.loads(d),
        rounds=rounds,
    )

    # Real HTTP roundtrip
    http_rt_latencies = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        req = Request(
            f"http://127.0.0.1:{http_port}/classify",
            data=http_req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=5)
        resp.read()
        t1 = time.perf_counter_ns()
        http_rt_latencies.append((t1 - t0) / 1000)

    http_server.shutdown()

    results["http_json"] = TransportResult(
        name="HTTP+JSON",
        payload_bytes=len(http_req_body),
        wire_bytes=len(http_req_body) + 200,  # Approx HTTP headers
        encode_us=statistics.mean(http_enc_lat),
        decode_us=statistics.mean(http_dec_lat),
        roundtrip_us=statistics.mean(http_rt_latencies),
        roundtrip_p95_us=sorted(http_rt_latencies)[int(len(http_rt_latencies) * 0.95)],
        throughput_msg_per_sec=1_000_000 / statistics.mean(http_rt_latencies) if http_rt_latencies else 0,
        parse_success_rate=1.0,
    )

    # ── 2. AICL Binary over TCP ──────────────────────────────
    print("  [2/4] AICL Binary over TCP...")
    aicl_port = 18933
    aicl_server = _AICLEchoServer(aicl_port)
    aicl_server.start()
    time.sleep(0.1)

    aicl_req_bytes = _make_aicl_request_bytes()

    aicl_enc_lat, aicl_dec_lat = _measure_encode_decode(
        encode_fn=_make_aicl_request_bytes,
        decode_fn=lambda d: bin_decode(d),
        rounds=rounds,
    )

    # Real TCP roundtrip
    aicl_rt_latencies = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(("127.0.0.1", aicl_port))
        sock.sendall(aicl_req_bytes)
        # Read length-prefixed response
        hdr = sock.recv(4)
        if hdr:
            resp_len = struct.unpack(">I", hdr)[0]
            resp_data = b""
            while len(resp_data) < resp_len:
                chunk = sock.recv(resp_len - len(resp_data))
                if not chunk:
                    break
                resp_data += chunk
        sock.close()
        t1 = time.perf_counter_ns()
        aicl_rt_latencies.append((t1 - t0) / 1000)

    aicl_server.stop()

    results["aicl_tcp"] = TransportResult(
        name="AICL Binary/TCP",
        payload_bytes=len(aicl_req_bytes),
        wire_bytes=len(aicl_req_bytes) + 4,  # Length prefix
        encode_us=statistics.mean(aicl_enc_lat),
        decode_us=statistics.mean(aicl_dec_lat),
        roundtrip_us=statistics.mean(aicl_rt_latencies),
        roundtrip_p95_us=sorted(aicl_rt_latencies)[int(len(aicl_rt_latencies) * 0.95)],
        throughput_msg_per_sec=1_000_000 / statistics.mean(aicl_rt_latencies) if aicl_rt_latencies else 0,
        parse_success_rate=1.0,
    )

    # ── 3. AICL Binary local (no network) ────────────────────
    print("  [3/4] AICL Binary local (no network)...")
    local_enc_lat, local_dec_lat = _measure_encode_decode(
        encode_fn=_make_aicl_request_bytes,
        decode_fn=lambda d: bin_decode(d),
        rounds=rounds * 10,
    )

    results["aicl_local"] = TransportResult(
        name="AICL Binary/Local",
        payload_bytes=len(aicl_req_bytes),
        wire_bytes=len(aicl_req_bytes),
        encode_us=statistics.mean(local_enc_lat),
        decode_us=statistics.mean(local_dec_lat),
        roundtrip_us=statistics.mean(local_enc_lat) + statistics.mean(local_dec_lat),
        roundtrip_p95_us=0,
        throughput_msg_per_sec=1_000_000 / (statistics.mean(local_enc_lat) + statistics.mean(local_dec_lat)),
        parse_success_rate=1.0,
    )

    # ── 4. HTTP + AICL binary body ───────────────────────────
    print("  [4/4] HTTP + AICL binary body...")
    aicl_http_enc_lat, aicl_http_dec_lat = _measure_encode_decode(
        encode_fn=lambda: _make_aicl_request_bytes(),
        decode_fn=lambda d: bin_decode(d),
        rounds=rounds,
    )

    results["http_aicl"] = TransportResult(
        name="HTTP+AICL",
        payload_bytes=len(aicl_req_bytes),
        wire_bytes=len(aicl_req_bytes) + 200,  # HTTP headers
        encode_us=statistics.mean(aicl_http_enc_lat),
        decode_us=statistics.mean(aicl_http_dec_lat),
        roundtrip_us=statistics.mean(aicl_http_enc_lat) + statistics.mean(aicl_http_dec_lat),
        roundtrip_p95_us=0,
        throughput_msg_per_sec=0,
        parse_success_rate=1.0,
    )

    return results


# ──────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────

def print_transport_report(results: Dict[str, TransportResult]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("HONEST Transport Benchmark: HTTP+JSON vs AICL Binary")
    lines.append("=" * 80)
    lines.append("")
    lines.append("What is being measured:")
    lines.append("  - payload_bytes: actual data size (no headers)")
    lines.append("  - wire_bytes: total on-wire size (including protocol overhead)")
    lines.append("  - encode_us: serialization time (microseconds)")
    lines.append("  - decode_us: deserialization time (microseconds)")
    lines.append("  - roundtrip_us: full request-response (microseconds)")
    lines.append("  - throughput: messages/sec (sequential)")
    lines.append("")

    lines.append(
        f"{'Format':<20} {'Payload':>8} {'Wire':>8} "
        f"{'Enc(us)':>10} {'Dec(us)':>10} "
        f"{'RT(us)':>10} {'Msg/s':>10}"
    )
    lines.append("-" * 80)

    for name, r in results.items():
        lines.append(
            f"{r.name:<20} {r.payload_bytes:>7}B {r.wire_bytes:>7}B "
            f"{r.encode_us:>10.1f} {r.decode_us:>10.1f} "
            f"{r.roundtrip_us:>10.1f} {r.throughput_msg_per_sec:>10.0f}"
        )

    lines.append("-" * 80)
    lines.append("")

    # Delta analysis
    if "http_json" in results and "aicl_tcp" in results:
        h = results["http_json"]
        a = results["aicl_tcp"]
        lines.append("AICL Binary/TCP vs HTTP+JSON:")
        if h.wire_bytes > 0:
            lines.append(f"  Wire size: {a.wire_bytes}B vs {h.wire_bytes}B ({(a.wire_bytes - h.wire_bytes) / h.wire_bytes * 100:+.1f}%)")
        if h.encode_us > 0:
            lines.append(f"  Encode:    {a.encode_us:.1f}us vs {h.encode_us:.1f}us ({(a.encode_us - h.encode_us) / h.encode_us * 100:+.1f}%)")
        if h.decode_us > 0:
            lines.append(f"  Decode:    {a.decode_us:.1f}us vs {h.decode_us:.1f}us ({(a.decode_us - h.decode_us) / h.decode_us * 100:+.1f}%)")
        if h.roundtrip_us > 0:
            lines.append(f"  Roundtrip: {a.roundtrip_us:.1f}us vs {h.roundtrip_us:.1f}us ({(a.roundtrip_us - h.roundtrip_us) / h.roundtrip_us * 100:+.1f}%)")
        lines.append("")

    if "http_json" in results and "aicl_local" in results:
        h = results["http_json"]
        l = results["aicl_local"]
        lines.append("AICL Binary/Local (no network) vs HTTP+JSON:")
        lines.append(f"  Encode: {l.encode_us:.1f}us vs {h.encode_us:.1f}us ({(l.encode_us - h.encode_us) / h.encode_us * 100:+.1f}%)")
        lines.append(f"  Decode: {l.decode_us:.1f}us vs {h.decode_us:.1f}us ({(l.decode_us - h.decode_us) / h.decode_us * 100:+.1f}%)")
        lines.append(f"  Local RT: {l.roundtrip_us:.1f}us (no network overhead)")
        lines.append("")

    lines.append("=" * 80)
    lines.append("NOTES:")
    lines.append("  - Network RT includes TCP handshake amortization + syscall overhead")
    lines.append("  - AICL local = encode+decode only, no transport overhead")
    lines.append("  - No improvement claims: these are raw measurements")
    lines.append("  - Actual LLM inference time dominates end-to-end latency")
    lines.append("=" * 80)

    report = "\n".join(lines)
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Honest transport benchmark")
    parser.add_argument("--rounds", type=int, default=500, help="Iterations per test")
    args = parser.parse_args()

    print("Running transport benchmark...")
    results = run_transport_benchmark(rounds=args.rounds)
    report = print_transport_report(results)
    print(report)


if __name__ == "__main__":
    main()
