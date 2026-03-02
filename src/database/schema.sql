-- =============================================================================
-- schema.sql
-- Full DDL for the Intelligent Risk Scoring & Monitoring Platform
-- Compatible with PostgreSQL 14+
--
-- Tables (star schema):
--   dim_user          — Customer dimension
--   dim_merchant      — Merchant dimension
--   fact_transaction  — Core transaction fact table
--   prediction_log    — Immutable model prediction audit log
--   model_registry    — Versioned model metadata store
--
-- Run order matters:
--   1. Extensions & enums
--   2. Dimension tables (no FKs to fact)
--   3. Fact table (FKs to dims)
--   4. Prediction log (FK to fact)
--   5. Model registry (standalone)
--   6. Indexes
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 0. Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";     -- trigram indexes for name search


-- ---------------------------------------------------------------------------
-- 1. ENUM Types
-- ---------------------------------------------------------------------------

DO $$ BEGIN
    CREATE TYPE risk_tier_enum AS ENUM ('low', 'medium', 'high', 'blocked');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE merchant_category_enum AS ENUM (
        'retail', 'food_beverage', 'travel', 'electronics',
        'gambling', 'crypto', 'healthcare', 'utilities', 'other'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE merchant_risk_level_enum AS ENUM ('low', 'medium', 'high');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE txn_status_enum AS ENUM (
        'pending', 'completed', 'declined', 'reversed', 'flagged'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE txn_channel_enum AS ENUM ('web', 'mobile', 'pos', 'atm', 'api');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE payment_method_enum AS ENUM (
        'credit_card', 'debit_card', 'bank_transfer', 'crypto', 'wallet', 'bnpl'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE prediction_outcome_enum AS ENUM (
        'true_positive', 'true_negative',
        'false_positive', 'false_negative', 'unknown'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE model_stage_enum AS ENUM (
        'candidate', 'staging', 'production', 'archived', 'failed'
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;


-- ---------------------------------------------------------------------------
-- 2. DIMENSION: dim_user
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_user (
    -- Surrogate key
    user_id             UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Business / natural key
    external_id         VARCHAR(64)     NOT NULL,

    -- Identity
    full_name           VARCHAR(255)    NOT NULL,
    email_hash          VARCHAR(128)    NOT NULL,   -- SHA-256 of normalised email
    country_code        CHAR(2)         NOT NULL,   -- ISO 3166-1 alpha-2

    -- Account attributes
    account_age_days    BIGINT          NOT NULL DEFAULT 0,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    kyc_verified        BOOLEAN         NOT NULL DEFAULT FALSE,

    -- Risk profile
    risk_tier           risk_tier_enum  NOT NULL DEFAULT 'medium',
    credit_score        NUMERIC(5,2),               -- Nullable; 300.00 – 850.00

    -- Denormalised aggregates (updated by ETL/trigger)
    lifetime_txn_count  BIGINT          NOT NULL DEFAULT 0,
    lifetime_txn_volume NUMERIC(18,4)   NOT NULL DEFAULT 0,

    -- Free-form notes
    notes               TEXT,

    -- Audit timestamps
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT ck_dim_user_age_positive    CHECK (account_age_days >= 0),
    CONSTRAINT ck_dim_user_credit_range    CHECK (credit_score IS NULL OR (credit_score >= 300 AND credit_score <= 850)),
    CONSTRAINT ck_dim_user_country_len     CHECK (char_length(country_code) = 2)
);

COMMENT ON TABLE  dim_user IS 'Customer dimension — star schema (SCD Type 1)';
COMMENT ON COLUMN dim_user.email_hash IS 'SHA-256 of lowercased, trimmed email — PII safe';
COMMENT ON COLUMN dim_user.lifetime_txn_volume IS 'Total lifetime spend in USD — denormalised for performance';


-- ---------------------------------------------------------------------------
-- 3. DIMENSION: dim_merchant
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS dim_merchant (
    -- Surrogate key
    merchant_id             UUID                        PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Business key
    external_id             VARCHAR(64)                 NOT NULL,

    -- Identity
    merchant_name           VARCHAR(255)                NOT NULL,
    category                merchant_category_enum      NOT NULL DEFAULT 'other',
    country_code            CHAR(2)                     NOT NULL,
    city                    VARCHAR(100),

    -- Channel
    is_online_only          BOOLEAN                     NOT NULL DEFAULT FALSE,
    is_active               BOOLEAN                     NOT NULL DEFAULT TRUE,

    -- Risk profile
    risk_level              merchant_risk_level_enum    NOT NULL DEFAULT 'medium',
    is_high_risk_category   BOOLEAN                     NOT NULL DEFAULT FALSE,

    -- Denormalised statistics (updated by ETL)
    historical_fraud_rate   NUMERIC(6,5)                NOT NULL DEFAULT 0.0,
    avg_transaction_amount  NUMERIC(12,4)               NOT NULL DEFAULT 0.0,
    total_txn_count         NUMERIC(18,0)               NOT NULL DEFAULT 0,

    -- Notes
    notes                   TEXT,

    -- Audit timestamps
    created_at              TIMESTAMPTZ                 NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ                 NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT ck_dim_merchant_fraud_rate       CHECK (historical_fraud_rate >= 0 AND historical_fraud_rate <= 1),
    CONSTRAINT ck_dim_merchant_avg_amount       CHECK (avg_transaction_amount >= 0),
    CONSTRAINT ck_dim_merchant_country_len      CHECK (char_length(country_code) = 2)
);

COMMENT ON TABLE  dim_merchant IS 'Merchant dimension — star schema (SCD Type 1)';
COMMENT ON COLUMN dim_merchant.historical_fraud_rate IS 'Rolling 90-day fraud rate [0.00000–1.00000] — updated by ETL';


-- ---------------------------------------------------------------------------
-- 4. FACT: fact_transaction  (immutable after status = completed/declined)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fact_transaction (
    -- Surrogate key
    transaction_id          UUID                    PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Idempotency
    external_txn_id         VARCHAR(128)            NOT NULL,

    -- Dimension foreign keys
    user_id                 UUID                    NOT NULL
                                REFERENCES dim_user(user_id) ON DELETE RESTRICT,
    merchant_id             UUID                    NOT NULL
                                REFERENCES dim_merchant(merchant_id) ON DELETE RESTRICT,

    -- Timing
    txn_timestamp           TIMESTAMPTZ             NOT NULL,

    -- Value
    amount                  NUMERIC(18,4)           NOT NULL,
    currency                CHAR(3)                 NOT NULL DEFAULT 'USD',
    amount_usd              NUMERIC(18,4)           NOT NULL,

    -- Classification
    status                  txn_status_enum         NOT NULL DEFAULT 'pending',
    channel                 txn_channel_enum        NOT NULL,
    payment_method          payment_method_enum     NOT NULL,

    -- Device & network (PII-hashed)
    ip_address_hash         VARCHAR(128),
    device_fingerprint_hash VARCHAR(128),

    -- Geo risk
    is_international        BOOLEAN                 NOT NULL DEFAULT FALSE,

    -- Ground truth label (set post-review)
    fraud_label             SMALLINT,
    labelled_at             TIMESTAMPTZ,
    review_notes            TEXT,

    -- Audit timestamps
    created_at              TIMESTAMPTZ             NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ             NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT ck_fact_txn_amount_positive          CHECK (amount > 0),
    CONSTRAINT ck_fact_txn_amount_usd_non_negative  CHECK (amount_usd >= 0),
    CONSTRAINT ck_fact_txn_fraud_label_binary       CHECK (fraud_label IS NULL OR fraud_label IN (0, 1)),
    CONSTRAINT ck_fact_txn_currency_len             CHECK (char_length(currency) = 3)
);

COMMENT ON TABLE  fact_transaction IS 'Core transaction fact table — immutable after completion';
COMMENT ON COLUMN fact_transaction.fraud_label IS '1=fraud, 0=legit, NULL=awaiting review';
COMMENT ON COLUMN fact_transaction.external_txn_id IS 'Upstream idempotency key for deduplication';


-- ---------------------------------------------------------------------------
-- 5. AUDIT: prediction_log  (append-only)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS prediction_log (
    -- Surrogate key
    log_id                  UUID                        PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Request identity
    request_id              UUID                        NOT NULL DEFAULT uuid_generate_v4(),
    batch_id                UUID,                       -- Groups batch prediction rows

    -- Transaction reference
    transaction_id          UUID                        NOT NULL
                                REFERENCES fact_transaction(transaction_id) ON DELETE RESTRICT,

    -- Model identity (denormalised — preserved even if registry row deleted)
    model_name              VARCHAR(128)                NOT NULL,
    model_version           VARCHAR(64)                 NOT NULL,

    -- Prediction output
    fraud_score             NUMERIC(6,5)                NOT NULL,
    risk_label              SMALLINT                    NOT NULL,
    decision_threshold      FLOAT                       NOT NULL DEFAULT 0.5,

    -- Feature snapshot
    feature_vector          JSONB,

    -- Performance
    latency_ms              INTEGER                     NOT NULL DEFAULT 0,

    -- Ground truth resolution
    outcome                 prediction_outcome_enum     NOT NULL DEFAULT 'unknown',
    outcome_resolved_at     TIMESTAMPTZ,

    -- Human review
    is_reviewed             BOOLEAN                     NOT NULL DEFAULT FALSE,
    reviewer_notes          TEXT,

    -- Audit timestamps
    created_at              TIMESTAMPTZ                 NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ                 NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT ck_pred_log_score_range      CHECK (fraud_score >= 0.0 AND fraud_score <= 1.0),
    CONSTRAINT ck_pred_log_label_binary     CHECK (risk_label IN (0, 1)),
    CONSTRAINT ck_pred_log_latency_pos      CHECK (latency_ms >= 0)
);

COMMENT ON TABLE  prediction_log IS 'Append-only audit log for all model predictions';
COMMENT ON COLUMN prediction_log.feature_vector IS 'JSONB snapshot of feature values at inference time';
COMMENT ON COLUMN prediction_log.outcome IS 'Resolved TP/TN/FP/FN once ground truth is known';


-- ---------------------------------------------------------------------------
-- 6. REGISTRY: model_registry
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS model_registry (
    -- Surrogate key
    registry_id             UUID                PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identity
    model_name              VARCHAR(128)        NOT NULL,
    model_version           VARCHAR(64)         NOT NULL,
    stage                   model_stage_enum    NOT NULL DEFAULT 'candidate',

    -- Artifact location
    artifact_path           VARCHAR(512)        NOT NULL,
    artifact_size_bytes     NUMERIC(18,0),

    -- Environment metadata
    framework               VARCHAR(64)         NOT NULL,
    framework_version       VARCHAR(32)         NOT NULL,
    python_version          VARCHAR(16)         NOT NULL,

    -- Training metadata
    training_rows           NUMERIC(18,0)       NOT NULL,
    training_duration_s     FLOAT,

    -- Snapshots
    feature_names           JSONB,
    hyperparameters         JSONB,
    metrics                 JSONB,

    -- Promotion state
    is_active               BOOLEAN             NOT NULL DEFAULT FALSE,
    promoted_at             TIMESTAMPTZ,
    promoted_by             VARCHAR(128),

    -- Notes
    description             TEXT,

    -- Audit timestamps
    created_at              TIMESTAMPTZ         NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ         NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT uq_model_registry_version       UNIQUE (model_name, model_version),
    CONSTRAINT ck_model_registry_training_rows CHECK (training_rows > 0)
);

COMMENT ON TABLE  model_registry IS 'Versioned model registry with promotion workflow';
COMMENT ON COLUMN model_registry.metrics IS 'JSONB: {auc_roc, auc_pr, f1, precision, recall, ...}';
COMMENT ON COLUMN model_registry.is_active IS 'Only one row per model_name should be TRUE at a time';


-- ---------------------------------------------------------------------------
-- 7. INDEXES
-- ---------------------------------------------------------------------------

-- dim_user
CREATE UNIQUE INDEX IF NOT EXISTS ix_dim_user_external_id      ON dim_user(external_id);
CREATE        INDEX IF NOT EXISTS ix_dim_user_country_risk     ON dim_user(country_code, risk_tier);
CREATE        INDEX IF NOT EXISTS ix_dim_user_kyc_active       ON dim_user(kyc_verified, is_active);

-- dim_merchant
CREATE UNIQUE INDEX IF NOT EXISTS ix_dim_merchant_external_id  ON dim_merchant(external_id);
CREATE        INDEX IF NOT EXISTS ix_dim_merchant_category     ON dim_merchant(category, risk_level);
CREATE        INDEX IF NOT EXISTS ix_dim_merchant_country      ON dim_merchant(country_code);
CREATE        INDEX IF NOT EXISTS ix_dim_merchant_fraud_rate   ON dim_merchant(historical_fraud_rate);

-- fact_transaction (critical — these drive window query performance)
CREATE UNIQUE INDEX IF NOT EXISTS ix_fact_txn_external_id      ON fact_transaction(external_txn_id);
CREATE        INDEX IF NOT EXISTS ix_fact_txn_user_time        ON fact_transaction(user_id, txn_timestamp DESC);
CREATE        INDEX IF NOT EXISTS ix_fact_txn_merchant_time    ON fact_transaction(merchant_id, txn_timestamp DESC);
CREATE        INDEX IF NOT EXISTS ix_fact_txn_fraud_label      ON fact_transaction(fraud_label, txn_timestamp DESC)
              WHERE fraud_label IS NOT NULL;  -- Partial index — skips unlabelled rows
CREATE        INDEX IF NOT EXISTS ix_fact_txn_status           ON fact_transaction(status);
CREATE        INDEX IF NOT EXISTS ix_fact_txn_timestamp        ON fact_transaction USING BRIN (txn_timestamp);  -- BRIN for time-series range scans

-- prediction_log
CREATE UNIQUE INDEX IF NOT EXISTS ix_pred_log_request_id       ON prediction_log(request_id);
CREATE        INDEX IF NOT EXISTS ix_pred_log_transaction_id   ON prediction_log(transaction_id);
CREATE        INDEX IF NOT EXISTS ix_pred_log_model_version    ON prediction_log(model_version, created_at DESC);
CREATE        INDEX IF NOT EXISTS ix_pred_log_outcome          ON prediction_log(outcome);
CREATE        INDEX IF NOT EXISTS ix_pred_log_batch_id         ON prediction_log(batch_id) WHERE batch_id IS NOT NULL;
CREATE        INDEX IF NOT EXISTS ix_pred_log_created_at       ON prediction_log USING BRIN (created_at);

-- model_registry
CREATE        INDEX IF NOT EXISTS ix_model_registry_name_stage ON model_registry(model_name, stage);
CREATE        INDEX IF NOT EXISTS ix_model_registry_active     ON model_registry(model_name, is_active)
              WHERE is_active = TRUE;  -- Partial index — only active models


-- ---------------------------------------------------------------------------
-- 8. AUTO-UPDATE updated_at trigger
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$ DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'dim_user', 'dim_merchant', 'fact_transaction',
        'prediction_log', 'model_registry'
    ]
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS trg_%s_updated_at ON %I;
             CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %I
             FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();',
            t, t, t, t
        );
    END LOOP;
END $$;
