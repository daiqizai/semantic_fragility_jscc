import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Dict, Optional


EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-S[0-6]-\d{3}$")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_metadata(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = run("rev-parse", "--verify", "HEAD")
    status = run("status", "--short")
    return {
        "commit": commit if commit else "UNBORN",
        "worktree_clean": not bool(status),
        "worktree_status": status.splitlines(),
    }


class _Tee:
    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.console.write(value)
        self.log_file.write(value)
        self.log_file.flush()
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()


class ExperimentRun:
    """Create a non-overwriting experiment directory and track its lifecycle."""

    def __init__(
        self,
        root: Path,
        config_path: Path,
        effective_config: Dict[str, Any],
    ):
        self.root = root
        self.config_path = config_path
        self.config = effective_config
        self.experiment_id = str(effective_config.get("experiment_id", ""))
        if not EXPERIMENT_ID_PATTERN.fullmatch(self.experiment_id):
            raise ValueError(
                "experiment_id must match EXP-S[0-6]-NNN, for example EXP-S1-001"
            )
        if not config_path.stem.startswith(self.experiment_id):
            raise ValueError(
                "The config filename must start with its experiment_id"
            )

        self.output_dir = root / "outputs" / self.experiment_id
        self.checkpoint_dir = root / "checkpoints" / self.experiment_id
        self.log_path = self.output_dir / "run.log"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.manifest_path = self.output_dir / "run_manifest.json"
        self._manifest: Dict[str, Any] = {}
        self._log_file = None
        self._stdout_redirect = None
        self._stderr_redirect = None

    def __enter__(self) -> "ExperimentRun":
        for path in (self.output_dir, self.checkpoint_dir):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite experiment artifacts: {path}. "
                    "Use a new experiment ID."
                )

        self.output_dir.mkdir(parents=True)
        self.checkpoint_dir.mkdir(parents=True)
        self.write_json(self.output_dir / "config.json", self.config)
        self._manifest = {
            "experiment_id": self.experiment_id,
            "status": "running",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "config_source": str(self.config_path.relative_to(self.root)),
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
            "git": git_metadata(self.root),
            "log_path": str(self.log_path.relative_to(self.root)),
            "metrics_path": str(self.metrics_path.relative_to(self.root)),
            "checkpoint_dir": str(self.checkpoint_dir.relative_to(self.root)),
            "error": None,
        }
        self.write_json(self.manifest_path, self._manifest)

        self._log_file = self.log_path.open("a", encoding="utf-8")
        tee_stdout = _Tee(sys.stdout, self._log_file)
        tee_stderr = _Tee(sys.stderr, self._log_file)
        self._stdout_redirect = contextlib.redirect_stdout(tee_stdout)
        self._stderr_redirect = contextlib.redirect_stderr(tee_stderr)
        self._stdout_redirect.__enter__()
        self._stderr_redirect.__enter__()
        print(
            f"experiment={self.experiment_id} "
            f"output={self.output_dir} checkpoint={self.checkpoint_dir}"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._manifest["status"] = "failed" if exc_value else "completed"
        self._manifest["finished_at_utc"] = utc_now()
        if exc_value:
            self._manifest["error"] = (
                f"{exc_type.__name__}: {exc_value}"
                if exc_type is not None
                else str(exc_value)
            )
            print(f"experiment failed: {self._manifest['error']}")
        else:
            print(f"experiment completed: {self.experiment_id}")
        self.write_json(self.manifest_path, self._manifest)

        if self._stderr_redirect is not None:
            self._stderr_redirect.__exit__(exc_type, exc_value, traceback)
        if self._stdout_redirect is not None:
            self._stdout_redirect.__exit__(exc_type, exc_value, traceback)
        if self._log_file is not None:
            self._log_file.close()
        return False

    def append_metrics(self, values: Dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, sort_keys=True) + "\n")

    @staticmethod
    def write_json(path: Path, values: Dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, sort_keys=True)
            handle.write("\n")

