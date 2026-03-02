# AutoML SaaS Platform

A No-Code Automated Machine Learning platform. Upload a CSV, train a model, get a prediction API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BROWSER (User)                              │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               React / Next.js Frontend                      │   │
│  │                                                             │   │
│  │  • Supabase Auth   → login / signup                        │   │
│  │  • Supabase Storage → direct CSV upload (no backend!)      │   │
│  │  • Supabase DB     → read project status, scores           │   │
│  │  • FastAPI         → POST /train  (trigger ML job)         │   │
│  │                      POST /predict/{id} (get predictions)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────────────┘
                         │  HTTP (REST)
              ┌──────────▼──────────┐
              │   FastAPI ML Worker  │
              │   (Pure ML Engine)   │
              │                     │
              │  1. Download CSV     │──────────────────────┐
              │  2. Preprocess       │                      │
              │  3. Auto-train 3     │              ┌───────▼──────────────┐
              │     models           │              │     Supabase          │
              │  4. Upload .joblib   │◄─────────────│                       │
              │  5. Update DB status │              │  • Auth (users)        │
              │  6. Serve /predict   │              │  • PostgreSQL (projects│
              └─────────────────────┘              │    prediction_logs)    │
                                                   │  • Storage             │
                                                   │    datasets/           │
                                                   │    models/             │
                                                   └───────────────────────┘
```

---

## Project Structure

---

## Quick Start

### 1. Supabase Setup
1. Create a new project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** → paste and run `supabase/init.sql`
3. Go to **Storage** → create two buckets: `datasets` (private) and `models` (private)

### 2. ML Service (FastAPI)
```bash
cd ml-service

# Copy and configure environment
cp .env.example .env
# Edit .env — set SUPABASE_URL and SUPABASE_SERVICE_KEY

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v

# Check health
curl http://localhost:8000/health/
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/detail
```

### 3. Using Docker Compose (recommended)
```bash
# From project root
cp ml-service/.env.example ml-service/.env
# Edit ml-service/.env

docker-compose up --build
```

---

## Environment Variables

### ML Service (`ml-service/.env`)

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | ✅ | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ✅ | Service role key (bypasses RLS — keep secret!) |
| `ENVIRONMENT` | ❌ | `development` \| `staging` \| `production` |
| `DATASETS_BUCKET` | ❌ | Storage bucket name for CSVs (default: `datasets`) |
| `MODELS_BUCKET` | ❌ | Storage bucket name for models (default: `models`) |
| `TRAINING_TIMEOUT_SEC` | ❌ | Max seconds per training job (default: `300`) |
| `MODEL_CACHE_SIZE` | ❌ | Models to keep hot in memory (default: `10`) |

### Frontend (`.env.local`)

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | ✅ | Same Supabase URL (safe to expose) |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | ✅ | Anon/public key (RLS-protected) |
| `NEXT_PUBLIC_FASTAPI_URL` | ✅ | URL of the FastAPI service |

---

## Build Roadmap

| Step | Status | Description |
|---|---|---|
| **Step 1** | ✅ Done | Project setup, Supabase init SQL, FastAPI boilerplate |
| **Step 2** | 🔜 Next | AutoML pipeline (preprocess → train → evaluate → upload) |
| **Step 3** | 🔜 | Frontend: Auth, CSV upload, column selector |
| **Step 4** | 🔜 | Prediction endpoint with model caching |
| **Step 5** | 🔜 | Frontend: Dashboard + prediction UI |
| **Step 6** | 🔜 | Production hardening (rate limiting, error recovery) |
