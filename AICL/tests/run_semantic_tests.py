"""Test runner for aicl.semantic tests (standalone, no pytest)."""
import sys
import traceback

sys.path.insert(0, ".")

TESTS = [
    ("tests.test_semantic", "test_action_roundtrip_dict"),
    ("tests.test_semantic", "test_action_validate_classify_requires_target"),
    ("tests.test_semantic", "test_action_validate_tool_requires_tool_name"),
    ("tests.test_semantic", "test_action_validate_confidence_range"),
    ("tests.test_semantic", "test_action_is_valid"),
    ("tests.test_semantic", "test_action_all_intents"),
    ("tests.test_semantic", "test_observation_roundtrip"),
    ("tests.test_semantic", "test_request_roundtrip"),
    ("tests.test_semantic", "test_result_roundtrip"),
    ("tests.test_semantic", "test_result_is_error"),
    ("tests.test_semantic", "test_adapter_action_to_bytes"),
    ("tests.test_semantic", "test_adapter_roundtrip_bytes"),
    ("tests.test_semantic", "test_adapter_action_to_sl"),
    ("tests.test_semantic", "test_adapter_result_to_bytes"),
    ("tests.test_semantic", "test_adapter_parse_json_output"),
    ("tests.test_semantic", "test_adapter_parse_empty_raises"),
    ("tests.test_semantic", "test_adapter_observation_to_request"),
    ("tests.test_semantic", "test_gate_rejects_outbound"),
    ("tests.test_semantic", "test_gate_rejects_inbound"),
    ("tests.test_semantic", "test_grammar_classify"),
    ("tests.test_semantic", "test_grammar_generate"),
    ("tests.test_semantic", "test_grammar_tool"),
    ("tests.test_semantic", "test_grammar_all_intents"),
    ("tests.test_semantic", "test_grammar_intents"),
    ("tests.test_semantic", "test_opcode_mapping_all_intents"),
    ("tests.test_semantic", "test_reverse_opcode_mapping"),
    ("tests.test_semantic", "test_classifier_opcodes"),
]

total = 0
passed = 0
failed = 0
errors = []

for module_name, fn_name in TESTS:
    total += 1
    try:
        import importlib
        mod = importlib.import_module(module_name)
        fn = getattr(mod, fn_name)
        fn()
        print(f"  PASS  {fn_name}")
        passed += 1
    except Exception as exc:
        print(f"  FAIL  {fn_name}: {exc}")
        failed += 1
        errors.append((module_name, fn_name, traceback.format_exc()))

print(f"\n=== Summary ===")
print(f"  Total:  {total}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")

if errors:
    print(f"\n=== Failure Details ===")
    for module, name, tb in errors:
        print(f"\n--- {module}.{name} ---")
        print(tb)

sys.exit(0 if failed == 0 else 1)
