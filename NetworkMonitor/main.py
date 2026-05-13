import argparse
import asyncio
import logging
from pathlib import Path

try:
    from .network_monitor import NetworkMonitor
except ImportError:  # pragma: no cover - supports `python NetworkMonitor/main.py`
    from network_monitor import NetworkMonitor


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Zeek-based ML network monitor.")
    parser.add_argument(
        "--models-dir",
        default=Path(__file__).resolve().parent / "models",
        type=Path,
        help="Directory containing exported models.",
    )
    parser.add_argument(
        "--mappings-path",
        default=REPO_ROOT / "DataPreprocessing" / "model_mappings" / "mappings.json",
        type=Path,
        help="Path to the label mapping JSON produced during preprocessing.",
    )
    parser.add_argument(
        "--log-dir",
        default=Path("/opt/zeek/logs/current"),
        type=Path,
        help="Directory containing Zeek conn.log.",
    )
    parser.add_argument(
        "--local-ip",
        default=None,
        help="Optional local IP address to ignore. Defaults to LOCAL_IP_ADDRESS.",
    )
    parser.add_argument(
        "--poll-interval",
        default=0.1,
        type=float,
        help="Seconds to wait between conn.log checks.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    network_monitor = NetworkMonitor(
        models_directory=args.models_dir,
        mappings_path=args.mappings_path,
        local_ip_address=args.local_ip,
    )
    await network_monitor.run(log_dir=args.log_dir, poll_interval=args.poll_interval)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Program exited cleanly.")
