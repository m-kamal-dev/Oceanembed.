"""
Computes the single source of truth for "is this app currently running on
real, validated observations, or on demo/synthetic data?" and writes it to
data_status.json. api_server.py reads this file directly -- it never
guesses or hardcodes this status.
"""
import json
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REAL_MODEL = ROOT / "model.pkl"
DEMO_MODEL = ROOT / "model_demo.pkl"
PROVENANCE_MANIFEST = ROOT / "data" / "dataset" / "provenance_manifest.json"
OUT = ROOT / "data_status.json"


def compute_status() -> dict:
    """Return the current real/demo/missing status dict, derived from
    whichever model artifact is actually present and loadable."""
    if REAL_MODEL.exists():
        try:
            with open(REAL_MODEL, "rb") as f:
                artifact = pickle.load(f)
            if artifact.get("data_mode") == "real":
                manifest = {}
                if PROVENANCE_MANIFEST.exists():
                    manifest = json.loads(PROVENANCE_MANIFEST.read_text())
                return {
                    "mode": "real",
                    "label": "Real Argo + Satellite Data (Validated)",
                    "n_train_rows": artifact.get("n_train_rows"),
                    "n_test_rows": artifact.get("n_test_rows"),
                    "wmo_count": manifest.get("argo_provenance", {}).get("unique_wmo_count"),
                    "validation_passed": manifest.get("validation_status") == "PASSED",
                }
        except (OSError, pickle.PickleError, json.JSONDecodeError) as e:
            logger.warning("Could not read %s (falling back to demo/missing check): %s", REAL_MODEL, e)

    if DEMO_MODEL.exists():
        return {
            "mode": "demo",
            "label": "Synthetic Demo Data — NOT real observations",
            "warning": "This build is running on formula-generated data for UI "
                       "testing only. Run scripts/build_dataset.py --use-raw on a "
                       "machine with internet access, then python train_model.py, "
                       "to switch to real validated data.",
        }

    return {
        "mode": "missing",
        "label": "No trained model found",
        "warning": "Run 'python train_model.py' (real data) or "
                   "'python train_model.py --demo' (synthetic) first.",
    }


if __name__ == "__main__":
    status = compute_status()
    OUT.write_text(json.dumps(status, indent=2))
    print(f"Wrote {OUT}: {status['mode']}")
