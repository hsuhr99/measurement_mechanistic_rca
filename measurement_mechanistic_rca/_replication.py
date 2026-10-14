"""Access to the bundled replication implementation."""

import importlib.util
from pathlib import Path
import sys


def ensure_replication_on_path() -> Path | None:
    if (
        importlib.util.find_spec("cali") is not None
        and importlib.util.find_spec("cali.cali") is not None
    ):
        return None

    root = Path(__file__).resolve().parents[1] / "replication_neurips" / "src"
    if not root.exists():
        raise ImportError(f"Bundled replication sources not found at {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ensure_topic_on_path() -> Path | None:
    if (
        importlib.util.find_spec("topic") is not None
        and importlib.util.find_spec("topic.topic") is not None
    ):
        return None

    root = Path(__file__).resolve().parents[1] / "replication_neurips" / "src" / "competitors" / "topic" / "src"
    if not root.exists():
        raise ImportError(f"Bundled TOPIC sources not found at {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def ensure_dodiscover_on_path() -> Path | None:
    if (
        importlib.util.find_spec("dodiscover") is not None
        and importlib.util.find_spec("dodiscover.toporder.nogam") is not None
    ):
        return None

    root = Path(__file__).resolve().parents[1] / "replication_neurips" / "src" / "competitors" / "dodiscover"
    if not root.exists():
        raise ImportError(f"Bundled dodiscover sources not found at {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
