-- =============================================================================
-- AutoML SaaS Platform — Supabase Initialization Script
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- SECTION 1: EXTENSIONS
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- For uuid_generate_v4()
CREATE EXTENSION IF NOT EXISTS "pgcrypto";     -- For gen_random_uuid() (pg 13+)


-- -----------------------------------------------------------------------------
-- SECTION 2: CORE TABLES
-- -----------------------------------------------------------------------------

-- projects: One row per ML training job initiated by a user.
-- This is the central record that tracks the full lifecycle of a model.
CREATE TABLE IF NOT EXISTS public.projects (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID            NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    -- Metadata provided by user
    name            TEXT            NOT NULL,
    description     TEXT,

    -- Dataset information (set when CSV is uploaded)
    dataset_path    TEXT,                          -- e.g. "datasets/{user_id}/{project_id}/data.csv"
    dataset_filename TEXT,                         -- Original filename shown in the UI
    target_column   TEXT,                          -- Column the model will predict
    feature_columns TEXT[],                        -- Remaining columns used as features

    -- ML training results (set by the FastAPI worker upon completion)
    task_type       TEXT CHECK (task_type IN ('classification', 'regression')),
    winning_model   TEXT,                          -- e.g. "RandomForestClassifier"
    accuracy_score  FLOAT,                         -- e.g. 0.9423 (primary metric)
    metric_name     TEXT,                          -- e.g. "f1_score" or "r2_score"
    all_scores      JSONB,                         -- Full results: {"RandomForest": 0.94, "LogisticReg": 0.91}
    model_path      TEXT,                          -- e.g. "models/{user_id}/{project_id}/model.joblib"

    -- Lifecycle status
    -- pending → training → completed | failed
    status          TEXT            NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'training', 'completed', 'failed')),
    error_message   TEXT,                          -- Populated if status = 'failed'

    -- Timestamps
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    trained_at      TIMESTAMPTZ                    -- When training completed
);

-- Automatically keep updated_at current on every row update
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_projects_updated
    BEFORE UPDATE ON public.projects
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();


-- prediction_logs: Audit log of every /predict call against a deployed model.
-- Useful for monitoring usage and debugging model behavior.
CREATE TABLE IF NOT EXISTS public.prediction_logs (
    id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID            NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,
    user_id         UUID            NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    input_data      JSONB           NOT NULL,      -- Raw JSON payload sent to /predict
    prediction      JSONB           NOT NULL,      -- Model output (value + probabilities if classification)
    model_version   TEXT,                          -- The winning_model name for traceability
    latency_ms      INTEGER,                       -- Inference time in milliseconds

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);


-- user_profiles: Extends auth.users with app-specific metadata.
-- Auto-created via trigger on new user signup.
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id              UUID            PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name       TEXT,
    avatar_url      TEXT,
    plan            TEXT            NOT NULL DEFAULT 'free'
                        CHECK (plan IN ('free', 'pro', 'enterprise')),
    projects_count  INTEGER         NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Auto-create a user_profile row whenever a new user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, full_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.raw_user_meta_data ->> 'full_name',
        NEW.raw_user_meta_data ->> 'avatar_url'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- -----------------------------------------------------------------------------
-- SECTION 3: INDEXES
-- Optimize the most common query patterns.
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_projects_user_id
    ON public.projects (user_id);

CREATE INDEX IF NOT EXISTS idx_projects_status
    ON public.projects (status);

CREATE INDEX IF NOT EXISTS idx_projects_user_status
    ON public.projects (user_id, status);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_project_id
    ON public.prediction_logs (project_id);

CREATE INDEX IF NOT EXISTS idx_prediction_logs_user_id
    ON public.prediction_logs (user_id);


-- -----------------------------------------------------------------------------
-- SECTION 4: ROW LEVEL SECURITY (RLS)
-- Critical: ensures users can ONLY see and modify their own data.
-- Never skip this in production.
-- -----------------------------------------------------------------------------

-- Enable RLS on all user-facing tables
ALTER TABLE public.projects        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prediction_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_profiles   ENABLE ROW LEVEL SECURITY;

-- ── projects policies ──────────────────────────────────────────────────────
-- Users can SELECT only their own projects
CREATE POLICY "users_select_own_projects"
    ON public.projects FOR SELECT
    USING (auth.uid() = user_id);

-- Users can INSERT only rows where user_id = their own id
CREATE POLICY "users_insert_own_projects"
    ON public.projects FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Users can UPDATE only their own projects
CREATE POLICY "users_update_own_projects"
    ON public.projects FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Users can DELETE only their own projects
CREATE POLICY "users_delete_own_projects"
    ON public.projects FOR DELETE
    USING (auth.uid() = user_id);

-- ── Service Role bypass (for FastAPI worker) ───────────────────────────────
-- The FastAPI server uses the SERVICE_ROLE key, which bypasses RLS by default.
-- No explicit policy needed — service_role bypasses all RLS automatically.
-- IMPORTANT: Never expose the service_role key to the frontend.

-- ── prediction_logs policies ───────────────────────────────────────────────
CREATE POLICY "users_select_own_logs"
    ON public.prediction_logs FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "users_insert_own_logs"
    ON public.prediction_logs FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- ── user_profiles policies ─────────────────────────────────────────────────
CREATE POLICY "users_select_own_profile"
    ON public.user_profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "users_update_own_profile"
    ON public.user_profiles FOR UPDATE
    USING (auth.uid() = id);


-- -----------------------------------------------------------------------------
-- SECTION 5: STORAGE BUCKETS
-- Create via Supabase Dashboard → Storage → New Bucket, OR via the API.
-- The SQL below documents the intent; actual bucket creation uses the JS client
-- or Dashboard UI. Policies below assume buckets are created.
-- -----------------------------------------------------------------------------

-- Run this block only if you have the storage schema available:
-- INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
-- VALUES
--   ('datasets', 'datasets', false, 52428800, ARRAY['text/csv', 'application/csv', 'text/plain']),
--   ('models',   'models',   false, 524288000, ARRAY['application/octet-stream']);

-- ── Storage RLS: datasets bucket ──────────────────────────────────────────
-- Users upload to their own namespaced folder: datasets/{user_id}/{project_id}/
CREATE POLICY "users_upload_own_datasets"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'datasets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "users_read_own_datasets"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'datasets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

CREATE POLICY "users_delete_own_datasets"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'datasets'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

-- ── Storage RLS: models bucket ─────────────────────────────────────────────
-- Models are written by the service role (FastAPI), read by authenticated users
CREATE POLICY "users_read_own_models"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'models'
        AND auth.uid()::text = (storage.foldername(name))[1]
    );

-- Service role (FastAPI) bypasses RLS — no insert policy needed for models.


-- -----------------------------------------------------------------------------
-- SECTION 6: HELPER VIEWS (optional but useful in dashboard)
-- -----------------------------------------------------------------------------

-- Quick summary view of a user's projects with key metrics
CREATE OR REPLACE VIEW public.project_summaries AS
SELECT
    p.id,
    p.user_id,
    p.name,
    p.status,
    p.task_type,
    p.winning_model,
    p.accuracy_score,
    p.metric_name,
    p.target_column,
    p.dataset_filename,
    p.created_at,
    p.trained_at,
    COUNT(pl.id) AS total_predictions
FROM public.projects p
LEFT JOIN public.prediction_logs pl ON pl.project_id = p.id
GROUP BY p.id;

-- RLS on the view (inherits from base table, but explicit is safer)
ALTER VIEW public.project_summaries OWNER TO authenticated;
