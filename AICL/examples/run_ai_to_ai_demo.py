"""
AICL AI-to-AI Demonstration — Python Launcher/Observer

This script launches the native Rust demo binary and monitors its output.
Python is ONLY used for launching/observing — it is NOT in the communication path.

Usage:
    python examples/run_ai_to_ai_demo.py

The actual AI-to-AI communication happens entirely in native Rust:
  Model A → Rust Runtime → SPSC Ring Buffer → Rust Runtime → Model B
"""

import subprocess
import sys
import os
from pathlib import Path


def find_demo_binary() -> str:
    """Locate the compiled demo binary."""
    project_root = Path(__file__).parent.parent
    candidate_dirs = [
        project_root / "core-rust" / "target" / "release" / "examples",
        project_root / "core-rust" / "target" / "debug" / "examples",
    ]
    for d in candidate_dirs:
        binary = d / "ai_to_ai_demo"
        if sys.platform == "win32":
            binary = d / "ai_to_ai_demo.exe"
        if binary.exists():
            return str(binary)
    return ""


def main():
    print("=" * 64)
    print("  AICL AI-to-AI NATIVE DEMONSTRATION — Python Launcher")
    print("  Python is ONLY used for launching. Communication is native Rust.")
    print("=" * 64)
    print()

    # Check for native binary
    binary = find_demo_binary()
    if not binary:
        print("  Demo binary not found. Building...")
        print()
        build_cmd = [
            "cargo", "+stable-x86_64-pc-windows-gnu",
            "build", "--release", "--example", "ai_to_ai_demo",
        ]
        print(f"  $ {' '.join(build_cmd)}")
        print()
        result = subprocess.run(
            build_cmd,
            cwd=str(Path(__file__).parent.parent / "core-rust"),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Build failed:\n{result.stderr}")
            sys.exit(1)
        binary = find_demo_binary()
        if not binary:
            print("  Binary not found after build.")
            sys.exit(1)

    print(f"  Binary: {binary}")
    print()

    # Run the demo
    print("  Launching native demo (Rust → Rust → Rust, no Python in path):")
    print("-" * 64)
    print()

    try:
        result = subprocess.run(
            [binary],
            cwd=str(Path(__file__).parent.parent / "core-rust"),
        )
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(f"  Error: Could not execute {binary}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
