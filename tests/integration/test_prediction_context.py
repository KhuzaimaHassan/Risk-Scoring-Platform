"""Exercise the real PostgreSQL scoring query on connection-local temporary tables."""
import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from src.features.feature_pipeline import FEATURE_COLUMNS, _featurise_single_row
from src.services.prediction_service import _fetch_scoring_context, _split_history


@pytest.mark.skipif(not os.environ.get('INTEGRATION_DATABASE_URL'), reason='Requires isolated PostgreSQL')
def test_single_read_context_preserves_past_only_feature_values():
    async def run():
        engine = create_async_engine(os.environ['INTEGRATION_DATABASE_URL'])
        now = datetime(2026, 2, 1, 12, tzinfo=timezone.utc)
        uid, other_uid, mid, other_mid = [uuid.uuid4() for _ in range(4)]
        target_id = uuid.uuid4()
        # Bounds, equal timestamps, unlabelled data and both kinds of entity match.
        cases = [
            (uid, mid, now-timedelta(days=30), 'COMPLETED', 10, 0, now-timedelta(days=29)),
            (uid, mid, now-timedelta(days=30,seconds=1), 'COMPLETED', 9999, 1, now-timedelta(days=29)),
            (uid, other_mid, now-timedelta(hours=1), 'COMPLETED', 20, None, None),
            (other_uid, mid, now-timedelta(hours=24), 'COMPLETED', 30, 1, now+timedelta(hours=1)),
            (uid, mid, now-timedelta(days=7), 'COMPLETED', 40, 1, now-timedelta(hours=1)),
            (uid, mid, now, 'COMPLETED', 9999, 1, now),
            (uid, mid, now+timedelta(seconds=1), 'COMPLETED', 9999, 1, now),
            (uid, mid, now-timedelta(minutes=1), 'FAILED', 9999, 1, now),
            (other_uid, other_mid, now-timedelta(minutes=1), 'COMPLETED', 9999, 1, now),
        ]
        rows = [dict(transaction_id=uuid.uuid4(),user_id=u,merchant_id=m,txn_timestamp=ts,
                     status=status,amount_usd=amount,amount=amount,currency='USD',
                     channel='MOBILE',payment_method='BUY_NOW_PAY_LATER',is_international=False,
                     fraud_label=label,labelled_at=labelled_at)
                for u,m,ts,status,amount,label,labelled_at in cases]
        target = dict(rows[0],transaction_id=target_id,txn_timestamp=now,amount_usd=100,
                      amount=100,fraud_label=None,labelled_at=None)
        try:
            async with engine.connect() as connection:
                async with connection.begin():
                    # TEMP relations shadow any permanent tables; rollback removes fixtures.
                    await connection.execute(text('''CREATE TEMP TABLE dim_user (
                        user_id uuid PRIMARY KEY, account_age_days int DEFAULT 30,
                        risk_tier text DEFAULT 'MEDIUM', credit_score int DEFAULT 670,
                        kyc_verified bool DEFAULT true, country_code text DEFAULT 'US') ON COMMIT DROP'''))
                    await connection.execute(text('''CREATE TEMP TABLE dim_merchant (
                        merchant_id uuid PRIMARY KEY, risk_level text DEFAULT 'HIGH',
                        historical_fraud_rate float DEFAULT .9, is_high_risk_category bool DEFAULT false,
                        is_online_only bool DEFAULT true, avg_transaction_amount float DEFAULT 99999,
                        category text DEFAULT 'retail', country_code text DEFAULT 'US') ON COMMIT DROP'''))
                    await connection.execute(text('''CREATE TEMP TABLE fact_transaction (
                        transaction_id uuid PRIMARY KEY, user_id uuid, merchant_id uuid,
                        txn_timestamp timestamptz, amount_usd numeric(18,4), amount numeric(18,4),
                        currency text, status text, channel text, payment_method text,
                        is_international bool, fraud_label int, labelled_at timestamptz) ON COMMIT DROP'''))
                    await connection.execute(text('INSERT INTO dim_user (user_id) VALUES (:id)'),[{'id':uid},{'id':other_uid}])
                    await connection.execute(text('INSERT INTO dim_merchant (merchant_id) VALUES (:id)'),[{'id':mid},{'id':other_mid}])
                    columns = list(target)
                    await connection.execute(text('INSERT INTO fact_transaction ('+','.join(columns)+') VALUES ('+
                                                  ','.join(':'+c for c in columns)+')'), rows+[target])
                    queries = []
                    def record(*args): queries.append(args[2])
                    event.listen(engine.sync_engine,'before_cursor_execute',record)
                    async with AsyncSession(bind=connection,expire_on_commit=False) as db:
                        context = await _fetch_scoring_context(db,target_id)
                        assert len(queries) == 1
                        assert context is not None
                        users, merchants = _split_history(context)
                        assert len(users) == 3 and len(merchants) == 3
                        expected = pd.DataFrame(rows)
                        expected = expected[(expected.txn_timestamp >= now-timedelta(days=30)) &
                                            (expected.txn_timestamp < now) & (expected.status=='COMPLETED')]
                        reference = _featurise_single_row(pd.Series(context),
                            expected[expected.user_id==uid],expected[expected.merchant_id==mid])
                        actual = _featurise_single_row(pd.Series(context),users,merchants)
                        np.testing.assert_allclose([actual[k] for k in FEATURE_COLUMNS],
                                                   [reference[k] for k in FEATURE_COLUMNS],rtol=1e-12)
                        assert actual['user_txn_count_1h'] == 1
                        assert actual['user_txn_count_30d'] == 3
                        assert actual['merchant_fraud_rate_7d'] == 1  # unavailable future label excluded
                        assert await _fetch_scoring_context(db,uuid.uuid4()) is None
                    event.remove(engine.sync_engine,'before_cursor_execute',record)
        finally:
            await engine.dispose()
    asyncio.run(run())
