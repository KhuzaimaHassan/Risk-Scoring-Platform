"""Authenticated end-to-end smoke check; never seeds or resets a database."""
import argparse
import os
import sys
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    password = os.environ.get("ADMIN_PASSWORD")
    if not password:
        raise SystemExit("Set ADMIN_PASSWORD for the service under test; credentials are never printed.")
    with httpx.Client(base_url=args.base_url, timeout=45) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200 and health.json()["status"] == "ok", "Readiness failed"
        schema = client.get("/openapi.json")
        schema.raise_for_status()
        assert "/api/v1/seed" not in schema.json()["paths"], "Reset route must be absent"
        assert client.get("/api/v1/dashboard/stats").status_code == 401, "Dashboard must require authentication"
        token = client.post("/api/v1/token", data={"username":os.environ.get("ADMIN_USERNAME","admin"),"password":password})
        token.raise_for_status()
        client.headers["Authorization"] = "Bearer " + token.json()["access_token"]
        assert client.post("/api/v1/predict",json={"transaction_id":"bad"}).status_code == 422
        assert client.post("/api/v1/predict",json={"transaction_id":"00000000-0000-4000-8000-000000000000"}).status_code == 404
        choices = client.get("/api/v1/dashboard/transactions").json()
        assert choices, "No stored transactions are available for scoring"
        tid = choices[0]["id"]
        predictions = []
        counts = []
        for _ in range(2):
            result = client.post("/api/v1/predict",json={"transaction_id":tid,"include_features":True})
            result.raise_for_status()
            score = result.json()
            assert 0 <= score["risk_score"] <= 100
            assert abs(score["risk_score"] - score["fraud_probability"]*100) < .011
            feed = client.get("/api/v1/dashboard/recent-transactions").json()
            entry = next(row for row in feed if row["log_id"] == score["log_id"])
            assert abs(entry["score"] - score["risk_score"]) < .011
            assert abs(entry["amount_usd"] - score["feature_snapshot"]["amount_usd"]) < .0001
            counts.append(client.get("/api/v1/dashboard/stats").json()["total_transactions"])
            predictions.append({"score":score["risk_score"],"latency_ms":score["latency_ms"],
                                "server_ms":result.headers.get("x-process-time-ms"),
                                "stages":result.headers.get("server-timing")})
        assert counts[0] == counts[1], "Repeated scoring inflated unique transaction KPI"
        batch = client.post("/api/v1/predict/batch",json={"transaction_ids":[tid,"00000000-0000-4000-8000-000000000000"]})
        batch.raise_for_status()
        assert batch.json()["total_scored"] == 1 and batch.json()["errors"][0]["status"] == 404
        assert client.get("/api/v1/models").status_code == 200
        assert client.get("/api/v1/dashboard/review-queue").status_code == 200
        print("PASS: readiness, reset removal, authentication, validation, scoring, durable feed, currency, unique KPIs, mixed batch, models, review queue")
        print("Prediction samples:", predictions)

if __name__ == "__main__":
    main()
