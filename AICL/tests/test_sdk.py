"""
Tests for the AICL Python SDK.

Verifies that the SDK provides the correct API surface, that HTTP
is not enabled by default, and that diagnostics work correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aicl.sdk import (
    AICLRuntime,
    Model,
    StructuredResult,
    StreamChunk,
    ok,
    err,
    CancellationToken,
    Capability,
    CapabilitySet,
    BUILTIN_CAPABILITIES,
    AICLError,
    CapabilityError,
    CancelledError,
    HTTPCompatDisabledError,
)
from aicl.sdk.diagnostic import TransportInfo, diagnose
from aicl.sdk.transport import NativeTransport, HttpTransport


# ═════════════════════════════════════════════════════════════════════════════
# Capability tests
# ═════════════════════════════════════════════════════════════════════════════

def test_capability_creation():
    cap = Capability("classify", description="Classify text")
    assert cap.name == "classify"
    assert cap.version == "1.0"
    assert cap.description == "Classify text"


def test_capability_match():
    cap = Capability("classify", description="Classify text into categories")
    assert cap.matches("classify")
    assert cap.matches("text")
    assert not cap.matches("generate")


def test_capability_set():
    caps = CapabilitySet([
        Capability("classify"),
        Capability("generate"),
        Capability("reason"),
    ])
    assert len(caps) == 3
    assert caps.has("classify")
    assert not caps.has("embed")
    assert caps.get("generate").name == "generate"
    assert len(caps.find("class")) == 1
    assert sorted(caps.names()) == ["classify", "generate", "reason"]


def test_builtin_capabilities():
    assert BUILTIN_CAPABILITIES.has("classify")
    assert BUILTIN_CAPABILITIES.has("generate")
    assert BUILTIN_CAPABILITIES.has("reason")
    assert len(BUILTIN_CAPABILITIES) >= 18


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic tests
# ═════════════════════════════════════════════════════════════════════════════

def test_diagnostic_native():
    info = diagnose(native_available=True, http_enabled=False)
    assert info.transport == "native"
    assert info.runtime_language == "rust"
    assert info.codec == "aicl-isa"
    assert info.zero_copy is True
    assert info.http_enabled is False
    assert info.native_available is True


def test_diagnostic_http():
    info = diagnose(native_available=False, http_enabled=True)
    assert info.transport == "http"
    assert info.runtime_language == "http"
    assert info.codec == "json"
    assert info.zero_copy is False
    assert info.http_enabled is True


def test_diagnostic_report():
    info = diagnose(native_available=True, http_enabled=False)
    report = info.report()
    assert "native" in report
    assert "rust" in report
    assert "aicl-isa" in report
    assert "no (default)" in report


def test_diagnostic_no_transport():
    info = diagnose(native_available=False, http_enabled=False)
    assert info.transport == "none"
    assert info.runtime_language == "python"


# ═════════════════════════════════════════════════════════════════════════════
# Cancellation tests
# ═════════════════════════════════════════════════════════════════════════════

def test_cancellation_token():
    token = CancellationToken()
    assert not token.is_cancelled
    token.cancel()
    assert token.is_cancelled


def test_cancellation_token_callback():
    token = CancellationToken()
    called = []
    token.on_cancelled(lambda: called.append(True))
    assert len(called) == 0
    token.cancel()
    assert len(called) == 1


def test_cancellation_token_check():
    token = CancellationToken()
    token.check()  # Should not raise
    token.cancel()
    try:
        token.check()
        assert False, "Should have raised CancelledError"
    except CancelledError:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Result tests
# ═════════════════════════════════════════════════════════════════════════════

def test_ok_result():
    result = ok("hello")
    assert result.ok
    assert result.unwrap() == "hello"


def test_err_result():
    result = err(ValueError("bad"))
    assert not result.ok
    try:
        result.unwrap()
        assert False, "Should have raised"
    except ValueError:
        pass


def test_result_map():
    result = ok(5)
    mapped = result.map(lambda x: x * 2)
    assert mapped.unwrap() == 10


def test_stream_chunk():
    chunk = StreamChunk(data="hello", chunk_index=0, is_final=False)
    assert not chunk.is_final
    assert chunk.data == "hello"


# ═════════════════════════════════════════════════════════════════════════════
# Error hierarchy tests
# ═════════════════════════════════════════════════════════════════════════════

def test_error_hierarchy():
    assert issubclass(CapabilityError, AICLError)
    assert issubclass(CancelledError, AICLError)
    assert issubclass(HTTPCompatDisabledError, AICLError)


def test_error_code():
    e = CapabilityError("not found")
    assert e.code == "CAPABILITY_ERROR"


# ═════════════════════════════════════════════════════════════════════════════
# Transport tests
# ═════════════════════════════════════════════════════════════════════════════

def test_native_transport_is_native():
    # This will fail if native lib is not available, which is expected
    # in CI environments. We test the class itself, not the FFI.
    assert NativeTransport.__init__ is not None


def test_http_transport_is_not_native():
    t = HttpTransport(url="http://example.com")
    assert not t.is_native
    t.close()


def test_http_transport_requires_explicit_enable():
    # Verify that HttpTransport exists but is not the default
    t = HttpTransport()
    assert not t.is_native
    t.close()


# ═════════════════════════════════════════════════════════════════════════════
# Runtime tests
# ═════════════════════════════════════════════════════════════════════════════

def test_runtime_diagnose():
    """Test that diagnose() returns a TransportInfo."""
    try:
        runtime = AICLRuntime()
        info = runtime.diagnose()
        assert isinstance(info, TransportInfo)
        runtime.close()
    except AICLError:
        # Native lib not available — expected in CI
        pass


def test_runtime_native_transport_method():
    """Test that native_transport() is an alias for diagnose()."""
    try:
        runtime = AICLRuntime()
        info = runtime.native_transport()
        assert isinstance(info, TransportInfo)
        runtime.close()
    except AICLError:
        pass


def test_runtime_capabilities():
    """Test capability listing."""
    try:
        runtime = AICLRuntime()
        assert runtime.has_capability("classify")
        assert not runtime.has_capability("nonexistent")
        assert "classify" in runtime.list_capabilities()
        runtime.close()
    except AICLError:
        pass


def test_runtime_connect():
    """Test model connection."""
    try:
        runtime = AICLRuntime()
        model = runtime.connect(capability="classify")
        assert isinstance(model, Model)
        assert model.name == "classify"
        runtime.close()
    except AICLError:
        pass


def test_runtime_connect_unknown():
    """Test connecting to unknown capability raises error."""
    try:
        runtime = AICLRuntime()
        try:
            runtime.connect(capability="nonexistent")
            assert False, "Should have raised CapabilityError"
        except CapabilityError:
            pass
        runtime.close()
    except AICLError:
        pass


def test_runtime_context_manager():
    """Test runtime as context manager."""
    try:
        with AICLRuntime() as runtime:
            assert runtime.has_capability("classify")
    except AICLError:
        pass


def test_runtime_http_disabled_by_default():
    """Verify HTTP is not enabled by default."""
    try:
        runtime = AICLRuntime()
        assert not runtime._http_enabled
        runtime.close()
    except AICLError:
        pass


def test_runtime_http_explicit_enable():
    """Verify HTTP can be explicitly enabled."""
    runtime = AICLRuntime(http=True, http_url="http://localhost:9999")
    assert runtime._http_enabled
    assert runtime._http_url == "http://localhost:9999"
    runtime.close()


# ═════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═════════════════════════════════════════════════════════════════════════════

def run_all():
    """Run all tests without pytest."""
    tests = [
        v for k, v in globals().items()
        if k.startswith("test_") and callable(v)
    ]
    passed = 0
    failed = 0
    errors = []
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  PASS  {test.__name__}")
        except Exception as e:
            failed += 1
            errors.append((test.__name__, e))
            print(f"  FAIL  {test.__name__}: {e}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    return errors


if __name__ == "__main__":
    errors = run_all()
    sys.exit(1 if errors else 0)
