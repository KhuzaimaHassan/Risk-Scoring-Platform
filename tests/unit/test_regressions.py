import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from config.settings import Settings
from src.main import create_app
from src.api.auth import create_access_token, verify_password, ADMIN_PASSWORD_HASH
from src.api.routes.auth import _attempts
from src.database.session import get_async_db
from src.features import feature_pipeline as fp
from src.features.feature_aggregation import compute_merchant_rolling_features
from src.features.feature_extraction import extract_raw_features
from src.services import prediction_service as ps
from src.services.model_state import ModelBundle


def test_production_rejects_default_or_missing_credentials():
    for password in ("", "admin123", "short"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, environment="production", secret_key=SecretStr("x"*40), admin_password=password)


def test_oversized_password_is_rejected_without_bcrypt_error():
    assert verify_password("x"*100, ADMIN_PASSWORD_HASH) is False
    assert verify_password("é"*40, ADMIN_PASSWORD_HASH) is False


@pytest.fixture
def client():
    _attempts.clear()
    app = create_app()
    app.state.model_bundle = None
    app.state.start_time = 0
    db = SimpleNamespace(execute=AsyncMock(), commit=AsyncMock(), rollback=AsyncMock())
    async def database():
        yield db
    app.dependency_overrides[get_async_db] = database
    client = TestClient(app, raise_server_exceptions=False)
    yield client, app, db
    client.close()
    _attempts.clear()


def authorization():
    return {"Authorization": "Bearer " + create_access_token({"sub": "admin"})}


def test_reset_route_is_absent(client):
    c, app, _ = client
    assert "/api/v1/seed" not in app.openapi()["paths"]
    assert c.get("/api/v1/seed").status_code == 404


def test_readiness_fails_but_liveness_succeeds(client):
    c, _, db = client
    db.execute.side_effect = RuntimeError("database unavailable")
    r = c.get("/api/v1/health")
    assert r.status_code == 503
    assert r.json()["model_loaded"] is False
    assert c.get("/api/v1/live").status_code == 200


def test_model_guard_returns_503_before_database_queries(client):
    c, _, db = client
    r = c.post("/api/v1/predict", headers=authorization(), json={"transaction_id": str(uuid.uuid4())})
    assert r.status_code == 503
    db.execute.assert_not_awaited()


def test_authentication_and_invalid_uuid(client):
    c, _, _ = client
    assert c.get("/api/v1/dashboard/stats").status_code == 401
    assert c.post("/api/v1/token", data={"username":"admin", "password":"admin123"}).status_code == 401
    assert c.post("/api/v1/token", data={"username":"admin", "password":"x"*100}).status_code == 401
    assert c.post("/api/v1/token", data={"username":"admin", "password":"TestOnly-LongPassword!"}).status_code == 200
    assert c.post("/api/v1/predict", headers=authorization(), json={"transaction_id":"bad"}).status_code == 422


def test_login_throttling(client):
    c, _, _ = client
    for _ in range(10):
        assert c.post("/api/v1/token", data={"username":"admin", "password":"wrong"}).status_code == 401
    assert c.post("/api/v1/token", data={"username":"admin", "password":"wrong"}).status_code == 429


def test_exception_response_does_not_leak_traceback(client):
    c, app, _ = client
    @app.get("/test-error")
    async def fail():
        raise RuntimeError("private-database-secret")
    r = c.get("/test-error")
    assert r.status_code == 500
    assert "private-database-secret" not in r.text and "Traceback" not in r.text
    assert r.json()["request_id"]


def test_cors_configuration_is_respected(client):
    c, _, _ = client
    r = c.options("/api/v1/live", headers={"Origin":"https://untrusted.example", "Access-Control-Request-Method":"GET"})
    assert "access-control-allow-origin" not in r.headers


def transaction_rows():
    uid, mid = uuid.uuid4(), uuid.uuid4()
    result = []
    for timestamp, amount, label in [("2026-01-01T00:00:00Z",100,0), ("2026-01-11T00:00:00Z",200,None),
                                      ("2026-01-11T00:00:00Z",300,1), ("2026-01-11T00:30:00Z",400,0)]:
        ts = pd.Timestamp(timestamp)
        result.append(dict(transaction_id=uuid.uuid4(), user_id=uid, merchant_id=mid,
            txn_timestamp=ts, amount_usd=amount, fraud_label=label, labelled_at=ts+timedelta(hours=1),
            is_international=False, channel="MOBILE", payment_method="BUY_NOW_PAY_LATER",
            user_account_age_days=10, user_risk_tier="HIGH", user_credit_score=700, user_kyc_verified=True,
            merchant_risk_level="HIGH", merchant_historical_fraud_rate=.99,
            merchant_avg_transaction_amount=99999, merchant_is_high_risk_category=False, merchant_is_online_only=True))
    return pd.DataFrame(result)


@contextmanager
def no_database():
    yield None


def test_training_serving_parity_sparse_unlabelled_and_same_time_history():
    frame = transaction_rows()
    with patch.object(fp,"sync_db_context",no_database), patch.object(fp,"_load_full_transaction_table",return_value=frame):
        X, y = fp.FeaturePipeline().build_training_dataset()
    targets = frame[frame.fraud_label.notna()]
    assert len(X) == len(y) == 3
    for index, (_, row) in enumerate(targets.iterrows()):
        history = frame[(frame.txn_timestamp < row.txn_timestamp) & (frame.txn_timestamp >= row.txn_timestamp-timedelta(days=30))]
        online = fp._enforce_column_order(pd.DataFrame([fp._featurise_single_row(row,history,history)]))
        np.testing.assert_allclose(X.iloc[index].to_numpy(float), online.iloc[0].to_numpy(float))
    assert X.iloc[1].user_txn_count_1h == 0  # equal timestamps and ten-day-old row excluded
    assert X.iloc[2].user_txn_count_1h == 2  # includes unlabelled history
    assert X.iloc[2].merchant_fraud_rate_7d == 0  # labels arrive after scoring time


def test_future_labels_and_mutable_merchant_statistics_do_not_leak():
    frame = transaction_rows()
    row = frame.iloc[3]
    history = frame.iloc[1:3]
    feats = fp._featurise_single_row(row,history,history)
    assert feats["merchant_avg_transaction_amount"] == 250
    assert feats["merchant_historical_fraud_rate"] == 0
    assert not set(fp.FEATURE_COLUMNS) & fp._EXCLUDED
    available = history.copy()
    available["labelled_at"] = row.txn_timestamp - timedelta(seconds=1)
    assert compute_merchant_rolling_features(available,row.txn_timestamp,400)["merchant_fraud_rate_7d"] == 1


def test_training_target_labels_respect_cutoff():
    frame = transaction_rows()
    with patch.object(fp,"sync_db_context",no_database), patch.object(fp,"_load_full_transaction_table",return_value=frame):
        X, y = fp.FeaturePipeline().build_training_dataset(until=datetime(2026,1,11,0,45,tzinfo=timezone.utc))
    assert len(X) == len(y) == 1


def test_enum_names_and_values_encode_identically():
    ts = datetime.now(timezone.utc)
    def features(channel, method, risk):
        return extract_raw_features({"txn_timestamp":ts,"channel":channel,"payment_method":method},
                                    {"risk_tier":risk}, {"risk_level":risk})
    assert features("MOBILE","BUY_NOW_PAY_LATER","HIGH") == features("mobile","bnpl","high")


def test_recommendation_follows_configured_threshold():
    assert ps._build_risk_context(.3, True, .2).recommended_action == "review"
    assert ps._build_risk_context(.8, False, .9).recommended_action == "allow"


def test_audit_failure_rolls_back_and_does_not_return_fake_id():
    async def run():
        db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
        service = ps.PredictionService(None,{},"test",.5)
        with patch.object(ps,"log_prediction",AsyncMock(side_effect=RuntimeError("write failed"))):
            with pytest.raises(RuntimeError,match="durably recorded"):
                await service._log_prediction_safe(db,uuid.uuid4(),.2,False,1,{})
        db.rollback.assert_awaited_once()
        db.commit.assert_not_awaited()
    asyncio.run(run())


def test_audit_commit_failure_never_returns_success():
    async def run():
        db = SimpleNamespace(commit=AsyncMock(side_effect=RuntimeError("commit failed")),rollback=AsyncMock())
        service = ps.PredictionService(None,{},"test",.5)
        with patch.object(ps,"log_prediction",AsyncMock(return_value=SimpleNamespace(log_id=uuid.uuid4()))):
            with pytest.raises(RuntimeError):
                await service._log_prediction_safe(db,uuid.uuid4(),.2,False,1,{})
        db.rollback.assert_awaited_once()
    asyncio.run(run())


def test_audit_commit_precedes_success():
    async def run():
        db = SimpleNamespace(commit=AsyncMock(),rollback=AsyncMock())
        log_id = uuid.uuid4()
        service = ps.PredictionService(None,{},"test",.5)
        with patch.object(ps,"log_prediction",AsyncMock(return_value=SimpleNamespace(log_id=log_id))):
            assert await service._log_prediction_safe(db,uuid.uuid4(),.2,False,1,{}) == log_id
        db.commit.assert_awaited_once()
    asyncio.run(run())


def test_empty_dashboard_returns_no_invented_data():
    from src.api.routes.dashboard import get_chart_data, get_recent_transactions
    async def run():
        db = SimpleNamespace(scalars=AsyncMock(return_value=SimpleNamespace(all=lambda:[])),
                             execute=AsyncMock(return_value=SimpleNamespace(all=lambda:[])))
        assert await get_chart_data(db) == []
        assert await get_recent_transactions(db) == []
    asyncio.run(run())


def test_missing_transaction_context_is_one_query():
    async def run():
        db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(fetchone=lambda:None)))
        assert await ps._fetch_scoring_context(db,uuid.uuid4()) is None
        db.execute.assert_awaited_once()
    asyncio.run(run())


def test_empty_context_history_produces_valid_features():
    row = transaction_rows().iloc[0].to_dict()
    row['history_rows'] = []
    users, merchants = ps._split_history(row)
    assert users.empty and merchants.empty
    features = fp._featurise_single_row(pd.Series(row),users,merchants)
    assert features['user_txn_count_30d'] == 0
    assert features['merchant_txn_count_7d'] == 0


def test_timing_header_includes_committed_audit_work(client):
    c, app, _ = client
    app.state.model_bundle = ModelBundle(None,{},'test',.5)
    result = ps.PredictionResult(uuid.uuid4(),.2,False,ps._build_risk_context(.2,False),
        'test',.5,uuid.uuid4(),17,datetime.now(timezone.utc),None,
        {'context_db':5.0,'features':2.0,'inference':3.0,'audit':7.0})
    with patch.object(ps.PredictionService,'predict',AsyncMock(return_value=result)):
        response = c.post('/api/v1/predict',headers=authorization(),json={'transaction_id':str(result.transaction_id)})
    assert response.status_code == 200
    assert response.json()['latency_ms'] == 17
    assert response.headers['server-timing'] == 'context_db;dur=5.000, features;dur=2.000, inference;dur=3.000, audit;dur=7.000'


def test_service_latency_includes_time_waiting_for_audit_commit():
    async def run():
        row = transaction_rows().iloc[0].to_dict()
        row.update(amount=100,history_rows=[])
        clock = [0.0]
        async def commit(): clock[0] += .25
        db = SimpleNamespace(commit=AsyncMock(side_effect=commit),rollback=AsyncMock())
        service = ps.PredictionService(SimpleNamespace(predict_proba=lambda frame:np.array([[.8,.2]])),{},'test',.5)
        with patch.object(ps,'_fetch_scoring_context',AsyncMock(return_value=row)), \
             patch.object(ps,'log_prediction',AsyncMock(return_value=SimpleNamespace(log_id=uuid.uuid4()))), \
             patch.object(ps.time,'perf_counter',side_effect=lambda:clock[0]):
            result = await service.predict(db,row['transaction_id'])
        assert result.latency_ms == 250
        assert result.timings_ms['audit'] == 250
        db.commit.assert_awaited_once()
    asyncio.run(run())


def test_incompatible_model_is_rejected(tmp_path):
    from src.services.model_state import load_bundle
    with patch("src.services.model_state.ModelRegistry") as registry, patch("src.services.model_state.load_model_with_metadata",return_value=(None,{})):
        registry.return_value.get_latest_model.return_value={"version":"old"}
        with pytest.raises(RuntimeError,match="Incompatible"):
            load_bundle(tmp_path)


def test_threshold_selection_enforces_validation_recall():
    from src.training.evaluate import find_operating_threshold, compute_binary_metrics
    y = np.array([0,1,0,1,0,1,0,1])
    probabilities = np.array([.1,.2,.3,.4,.5,.6,.7,.8])
    threshold = find_operating_threshold(y, probabilities, .75)
    assert compute_binary_metrics(y,probabilities,threshold)["recall"] >= .75


def test_training_does_not_promote_a_model_that_fails_quality_gates(tmp_path):
    from src.training.train import train, TrainConfig
    X = pd.DataFrame({name: np.arange(200, dtype=float) % 7 for name in fp.FEATURE_COLUMNS})
    y = pd.Series(np.arange(200) % 2)
    cfg = TrainConfig(n_estimators=5, n_jobs=1, oob_score=False, auto_promote=True,
                      min_f1=1.0, min_recall=1.0, model_artifact_dir=tmp_path,
                      report_dir=tmp_path)
    with patch.object(fp.FeaturePipeline,"build_training_dataset",return_value=(X,y)):
        result = train(cfg, version="gate-test")
    assert result["quality_gate_passed"] is False
    assert result["promoted"] is False
    from src.training.model_registry import ModelRegistry
    assert ModelRegistry(tmp_path).summary()["current_production_model"] is None


def test_old_tokens_are_invalid_after_auth_key_change(client):
    import jwt
    from src.api.auth import settings
    c, _, _ = client
    legacy_token = jwt.encode({"sub":"admin", "exp":datetime.now(timezone.utc)+timedelta(minutes=5)},
                              settings.secret_key.get_secret_value(), algorithm=settings.jwt_algorithm)
    assert c.get("/api/v1/dashboard/stats", headers={"Authorization":"Bearer "+legacy_token}).status_code == 401


def test_service_import_does_not_require_importing_application_first():
    import subprocess, sys
    result = subprocess.run([sys.executable,"-c","from src.services.prediction_service import PredictionService; from src.services.model_state import load_bundle"],capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("reload_fails", [False, True])
def test_guarded_retraining_refreshes_serving_bundle_or_rolls_back(tmp_path, reload_fails):
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock
    from src.services import drift_service as drift
    from src.database.models import PredictionOutcome
    old = ModelBundle(SimpleNamespace(predict_proba=lambda X: np.array([[.9,.1],[.9,.1]])),{},"old",.5)
    new = ModelBundle(object(),{},"new",.2)
    app = SimpleNamespace(state=SimpleNamespace(model_bundle=old,last_retraining_attempt=None))
    settings = SimpleNamespace(retraining_cooldown_hours=24,monitoring_window_hours=24,
        performance_f1_min_threshold=.7,retraining_min_samples=10000,
        model_artifact_dir=tmp_path,model_report_dir=tmp_path,model_min_f1=.2,
        model_min_recall=.5,retraining_auto_promote=True)
    db = SimpleNamespace(execute=AsyncMock(return_value=SimpleNamespace(all=lambda:[
        (PredictionOutcome.TRUE_POSITIVE,1),(PredictionOutcome.FALSE_NEGATIVE,40)])),
        scalar=AsyncMock(return_value=12000))
    @asynccontextmanager
    async def factory():
        yield db
    result = {"version":"new","quality_gate_passed":True,"metrics":{"f1":.5,"recall":.7},
              "_evaluation_data":(pd.DataFrame({"a":[0,1]}),pd.Series([0,1]))}
    registry = MagicMock()
    loader = MagicMock(side_effect=RuntimeError("reload failed")) if reload_fails else MagicMock(return_value=new)
    with patch.object(drift,"get_settings",return_value=settings), patch.object(drift,"reconcile_ground_truth",AsyncMock()), \
         patch.object(drift,"AsyncSessionFactory",factory), patch.object(drift,"train",return_value=result), \
         patch.object(drift,"load_bundle",loader), patch("src.training.model_registry.ModelRegistry",return_value=registry):
        asyncio.run(drift.check_drift_and_retrain(app))
    assert app.state.model_bundle is (old if reload_fails else new)
    assert registry.promote_model.call_args_list[0].args[0] == "new"
    if reload_fails:
        assert registry.promote_model.call_args_list[-1].args[0] == "old"


def test_retraining_cooldown_prevents_repeated_training():
    from src.services import drift_service as drift
    app = SimpleNamespace(state=SimpleNamespace(model_bundle=object(),last_retraining_attempt=datetime.now(timezone.utc)))
    with patch.object(drift,"get_settings",return_value=SimpleNamespace(retraining_cooldown_hours=24)), \
         patch.object(drift,"reconcile_ground_truth",AsyncMock()), patch.object(drift,"train") as training:
        asyncio.run(drift.check_drift_and_retrain(app))
    training.assert_not_called()


def test_smoke_check_refuses_legacy_reset_route_without_requesting_it(monkeypatch):
    import importlib.util
    import sys
    import httpx
    spec = importlib.util.spec_from_file_location("smoke_check", Path(__file__).resolve().parents[2] / "scripts" / "test_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = []
    class LegacyClient:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, path):
            paths.append(path)
            assert path != "/api/v1/seed", "Never invoke the destructive legacy route"
            body = {"status":"ok"} if path.endswith("health") else {"paths":{"/api/v1/seed":{}}}
            return httpx.Response(200, json=body, request=httpx.Request("GET", "http://test"+path))
    monkeypatch.setattr(sys,"argv",["test_api.py"])
    monkeypatch.setattr(module.httpx,"Client",lambda **kwargs: LegacyClient())
    with pytest.raises(AssertionError,match="Reset route must be absent"):
        module.main()
    assert paths == ["/api/v1/health","/openapi.json"]
