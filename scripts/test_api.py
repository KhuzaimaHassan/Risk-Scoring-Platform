"""
scripts/test_api.py
--------------------
Quick smoke-test for the FastAPI inference layer.
Runs all key endpoints, prints results, and exits.

Usage:
    python scripts/test_api.py
"""
import sys
import time
import json

sys.path.insert(0, ".")

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)

BASE = "http://localhost:8000/api/v1"
TIMEOUT = 15  # seconds per request

# A known fraud transaction ID seeded in the DB
FRAUD_TXN_ID = "c1212b7d-809e-416a-9983-6001405ac382"

PASS = "[PASS]"
FAIL = "[FAIL]"

def check(label, response):
    ok = 200 <= response.status_code < 300
    status = PASS if ok else FAIL
    print(f"\n{status} {label}")
    print(f"     HTTP {response.status_code}")
    try:
        body = response.json()
        # Pretty print, truncate long fields
        for k, v in body.items():
            val = str(v)
            print(f"     {k}: {val[:120]}")
    except Exception:
        print(f"     (non-JSON) {response.text[:200]}")
    return ok


def main():
    print("=" * 55)
    print("  Risk Scoring Platform — API Smoke Test")
    print("=" * 55)

    results = []

    # ── 1. Health ─────────────────────────────────────────
    try:
        r = requests.get(f"{BASE}/health", timeout=TIMEOUT)
        results.append(check("GET /health", r))
    except requests.exceptions.ConnectionError:
        print(f"\n{FAIL} GET /health  ->  Connection refused (server not running?)")
        results.append(False)
    except requests.exceptions.Timeout:
        print(f"\n{FAIL} GET /health  ->  Timed out after {TIMEOUT}s")
        results.append(False)

    # ── 2. Model info ──────────────────────────────────────
    try:
        r = requests.get(f"{BASE}/model-info", timeout=TIMEOUT)
        results.append(check("GET /model-info", r))
    except Exception as e:
        print(f"\n{FAIL} GET /model-info  ->  {e}")
        results.append(False)

    # ── 3. Predict (known fraud transaction) ──────────────
    try:
        r = requests.post(
            f"{BASE}/predict",
            json={"transaction_id": FRAUD_TXN_ID},
            timeout=TIMEOUT,
        )
        results.append(check("POST /predict", r))
    except Exception as e:
        print(f"\n{FAIL} POST /predict  ->  {e}")
        results.append(False)

    # ── Summary ────────────────────────────────────────────
    print("\n" + "=" * 55)
    passed = sum(results)
    total  = len(results)
    print(f"  Result: {passed}/{total} endpoints passed")
    print("=" * 55)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
