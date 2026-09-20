"""``python -m anvira_runtime`` — daemon entry and CLI passthrough."""
import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "daemon":
        from .daemon import run_daemon
        args = sys.argv[2:]
        port = int(args[args.index("--port") + 1]) if "--port" in args else None
        grace = float(args[args.index("--idle-grace") + 1]) if "--idle-grace" in args else None
        return run_daemon(console_log="--console" in args, port_override=port,
                          auto_stop=True if "--auto-stop" in args else None, idle_grace_s=grace)
    from .cli.main import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
