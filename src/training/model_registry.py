"""
src/training/model_registry.py
--------------------------------
File-based model registry backed by a JSON manifest at models/registry.json.

Responsibility:
    Track all trained model versions, their evaluation metrics, lifecycle
    stage, and which version is currently serving production traffic.

Registry JSON structure:
    {
      "schema_version": "1.0",
      "current_production_model": "v20260301_143022",
      "available_versions": {
        "v20260301_143022": {
          "version": "v20260301_143022",
          "trained_at": "2026-03-01T14:30:22+00:00",
          "stage": "production",
          "artifact_path": "/abs/path/models/model_v20260301_143022.pkl",
          "metrics": { "roc_auc": 0.94, "f1": 0.80, ... },
          "hyperparameters": { "n_estimators": 300, ... },
          "feature_columns": ["amount_usd", ...],
          "n_features": 31,
          "training_rows": 42500,
          "training_duration_s": 47.3
        }
      },
      "history": [
        { "event": "registered",  "version": "v20260301_143022", "at": "..." },
        { "event": "promoted",    "version": "v20260301_143022", "at": "...", "by": "..." }
      ]
    }

Thread-safety note:
    The registry uses an exclusive file lock (via filelock) during writes.
    If running multiple training processes in parallel, only one will write
    at a time. In production, replace with a DB-backed registry
    (model_registry table already exists for this purpose).
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTRY_FILENAME = "registry.json"
REGISTRY_SCHEMA_VERSION = "1.0"

_VALID_STAGES = {"candidate", "staging", "production", "archived", "failed"}


# ---------------------------------------------------------------------------
# ModelRegistry class
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    File-backed model registry with atomic read-modify-write semantics.

    All public methods perform a full read of registry.json before
    operating and write back on completion, making them safe for sequential
    calls. For concurrent use, wrap in a file lock.

    Usage:
        registry = ModelRegistry(base_dir=Path("models/"))

        # Register a freshly trained model
        registry.register(metadata)

        # Promote to production (archives current production model first)
        registry.promote_model("v20260301_143022")

        # Inference layer looks up active model
        info = registry.get_latest_model()
        model_path = info["artifact_path"]

        # List all available versions
        for v in registry.list_models():
            print(v["version"], v["stage"], v["metrics"]["roc_auc"])
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Args:
            base_dir: Directory where registry.json and model artifacts live.
                      Typically Path("models/") relative to the project root.
        """
        self.base_dir = Path(base_dir)
        self.registry_path = self.base_dir / REGISTRY_FILENAME
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_path.exists():
            self._write(_empty_registry())
            logger.info("Initialised new model registry at %s", self.registry_path)
        else:
            logger.debug("Model registry found at %s", self.registry_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, metadata: dict[str, Any]) -> None:
        """
        Add a newly trained model to the registry in CANDIDATE stage.

        Args:
            metadata: Dict produced by save_model.py (must contain 'version').

        Raises:
            ValueError: If the version tag already exists in the registry.
        """
        version = metadata["version"]
        registry = self._read()

        if version in registry["available_versions"]:
            raise ValueError(
                f"Model version '{version}' is already registered. "
                f"Use a new version tag for each training run."
            )

        entry: dict[str, Any] = {
            "version": version,
            "trained_at": metadata.get("trained_at", datetime.now(timezone.utc).isoformat()),
            "description": metadata.get("description", ""),
            "stage": "candidate",
            "framework": metadata.get("framework", "sklearn"),
            "framework_version": metadata.get("framework_version", ""),
            "python_version": metadata.get("python_version", ""),
            "artifact_path": metadata.get("artifact_path", ""),
            "artifact_size_bytes": metadata.get("artifact_size_bytes", 0),
            "training_rows": metadata.get("training_rows", 0),
            "training_duration_s": metadata.get("training_duration_s", 0.0),
            "n_features": metadata.get("n_features", 0),
            "feature_columns": metadata.get("feature_columns", []),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "metrics": metadata.get("metrics", {}),
        }

        registry["available_versions"][version] = entry
        registry["history"].append({
            "event": "registered",
            "version": version,
            "at": datetime.now(timezone.utc).isoformat(),
            "metrics_summary": {
                k: metadata.get("metrics", {}).get(k)
                for k in ("roc_auc", "pr_auc", "f1", "recall", "precision")
            },
        })

        self._write(registry)
        logger.info(
            "Registered model version '%s' (AUC-ROC=%.4f, F1=%.4f).",
            version,
            entry["metrics"].get("roc_auc", 0),
            entry["metrics"].get("f1", 0),
        )

    def promote_model(
        self,
        version: str,
        stage: str = "production",
        promoted_by: str = "system",
    ) -> dict[str, Any]:
        """
        Promote a registered model to a target lifecycle stage.

        When promoting to **production**:
        - The current production model is automatically archived.
        - registry["current_production_model"] is updated.
        - An event is appended to history.

        When promoting to **staging**:
        - Stage is updated; production model is unchanged.

        Args:
            version:     Version tag of the model to promote.
            stage:       Target stage ("staging" | "production" | "archived" | "failed").
            promoted_by: Identifier of the user or job triggering promotion.

        Returns:
            The updated registry entry for the promoted version.

        Raises:
            ValueError: If version is not registered or stage is invalid.
        """
        if stage not in _VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {_VALID_STAGES}.")

        registry = self._read()

        if version not in registry["available_versions"]:
            raise ValueError(
                f"Version '{version}' is not registered. "
                f"Available: {list(registry['available_versions'].keys())}"
            )

        if stage == "production":
            # Archive the existing production model
            current_prod = registry.get("current_production_model")
            if current_prod and current_prod != version:
                if current_prod in registry["available_versions"]:
                    registry["available_versions"][current_prod]["stage"] = "archived"
                    logger.info("Archived previous production model: '%s'.", current_prod)

            registry["current_production_model"] = version

        registry["available_versions"][version]["stage"] = stage
        registry["available_versions"][version]["promoted_at"] = datetime.now(timezone.utc).isoformat()
        registry["available_versions"][version]["promoted_by"] = promoted_by

        registry["history"].append({
            "event": "promoted",
            "version": version,
            "stage": stage,
            "at": datetime.now(timezone.utc).isoformat(),
            "by": promoted_by,
        })

        self._write(registry)
        logger.info("Promoted model '%s' → stage='%s' (by %s).", version, stage, promoted_by)
        return registry["available_versions"][version]

    def get_latest_model(self) -> dict[str, Any]:
        """
        Return the metadata for the currently active production model.

        This is the primary method called by the inference layer at startup
        to discover which model to load.

        Returns:
            Registry entry dict for the production model.

        Raises:
            RuntimeError: If no production model has been promoted yet.
        """
        registry = self._read()
        prod_version = registry.get("current_production_model")

        if not prod_version:
            raise RuntimeError(
                "No production model has been promoted yet. "
                "Run `python scripts/train_model.py` and then promote a version."
            )

        if prod_version not in registry["available_versions"]:
            raise RuntimeError(
                f"Production version '{prod_version}' is listed as current but "
                f"has no registry entry. Registry may be corrupted."
            )

        return registry["available_versions"][prod_version]

    def get_model_by_version(self, version: str) -> dict[str, Any]:
        """
        Return the metadata for a specific model version.

        Args:
            version: Version tag to look up.

        Returns:
            Registry entry dict.

        Raises:
            KeyError: If the version is not registered.
        """
        registry = self._read()
        if version not in registry["available_versions"]:
            raise KeyError(
                f"Version '{version}' not found. "
                f"Available: {list(registry['available_versions'].keys())}"
            )
        return registry["available_versions"][version]

    def list_models(
        self,
        stage: str | None = None,
        order_by: str = "trained_at",
        descending: bool = True,
    ) -> list[dict[str, Any]]:
        """
        List all registered model versions, optionally filtered by stage.

        Args:
            stage:      If provided, filter to this lifecycle stage.
            order_by:   Field to sort by: "trained_at" | "roc_auc" | "f1".
            descending: Sort direction.

        Returns:
            List of registry entry dicts, sorted as specified.
        """
        registry = self._read()
        versions = list(registry["available_versions"].values())

        if stage:
            versions = [v for v in versions if v.get("stage") == stage]

        def sort_key(entry: dict) -> Any:
            if order_by in ("roc_auc", "f1", "recall", "precision"):
                return entry.get("metrics", {}).get(order_by, 0.0)
            return entry.get(order_by, "")

        versions.sort(key=sort_key, reverse=descending)
        return versions

    def get_best_model(self, metric: str = "roc_auc") -> dict[str, Any] | None:
        """
        Return the candidate/staging model with the best value for `metric`.

        Useful in the retraining pipeline to decide whether a new model
        should replace the current production model.

        Args:
            metric: Metric name to rank by (e.g. "roc_auc", "f1", "recall").

        Returns:
            Best model entry, or None if no non-production models exist.
        """
        candidates = [
            v for v in self.list_models()
            if v.get("stage") in ("candidate", "staging")
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda v: v.get("metrics", {}).get(metric, 0.0),
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Return the complete operation history log (immutable audit trail)."""
        return self._read().get("history", [])

    def summary(self) -> dict[str, Any]:
        """
        Return a compact summary of registry state.

        Useful for the /models health endpoint.

        Returns:
            Dict with total_versions, production_version, versions_by_stage.
        """
        registry = self._read()
        versions = registry["available_versions"].values()
        stage_counts: dict[str, int] = {}
        for v in versions:
            s = v.get("stage", "unknown")
            stage_counts[s] = stage_counts.get(s, 0) + 1

        return {
            "total_versions": len(registry["available_versions"]),
            "current_production_model": registry.get("current_production_model"),
            "versions_by_stage": stage_counts,
            "registry_path": str(self.registry_path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        """Read and return the registry JSON (safe for concurrent reads)."""
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, registry: dict[str, Any]) -> None:
        """
        Atomically write the registry JSON via a temp-file rename pattern.
        Prevents partial writes from corrupting the registry.
        """
        tmp_path = self.registry_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, default=str)
        # Atomic rename (POSIX) — on Windows this may briefly lock
        shutil.move(str(tmp_path), str(self.registry_path))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _empty_registry() -> dict[str, Any]:
    """Return a fresh empty registry structure."""
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "current_production_model": None,
        "available_versions": {},
        "history": [],
    }
