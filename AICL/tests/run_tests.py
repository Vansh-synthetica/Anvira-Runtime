"""Simple test runner for AICL-BIN codec tests (no pytest dependency)."""
import sys
import traceback
import importlib

TESTS = [
    "tests.test_codec",
    "tests.test_bin_errors",
    "tests.test_sl",
    "tests.test_semantic",
]


def discover_tests(module_name):
    """Yield (name, callable) for every test_* function in module."""
    mod = importlib.import_module(module_name)
    for name in dir(mod):
        if name.startswith("test_"):
            yield name, getattr(mod, name)


def main():
    total = 0
    passed = 0
    failed = 0
    errors = []

    sys.path.insert(0, ".")

    for module_name in TESTS:
        print(f"\n=== {module_name} ===")
        for name, fn in discover_tests(module_name):
            total += 1
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except Exception as exc:
                print(f"  FAIL  {name}: {exc}")
                failed += 1
                errors.append((module_name, name, traceback.format_exc()))

    print(f"\n=== Summary ===")
    print(f"  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if errors:
        print(f"\n=== Failure Details ===")
        for module, name, tb in errors:
            print(f"\n--- {module}.{name} ---")
            print(tb)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())