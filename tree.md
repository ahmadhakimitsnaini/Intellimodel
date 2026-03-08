.
├── README.md
├── frontend
│   ├── package.json
│   └── src
│       ├── app
│       │   ├── auth
│       │   ├── dashboard
│       │   │   ├── layout.tsx
│       │   │   └── page.tsx
│       │   ├── globals.css
│       │   ├── layout.tsx
│       │   └── public
│       ├── components
│       │   ├── dashboard
│       │   │   └── AppShell.tsx
│       │   └── ui
│       │       ├── EmptyState.tsx
│       │       ├── MetricCard.tsx
│       │       ├── ModelComparisonBar.tsx
│       │       ├── PageLoader.tsx
│       │       ├── ScoreRing.tsx
│       │       ├── Spinner.tsx
│       │       ├── StatusBadge.tsx
│       │       ├── TrainingPulse.tsx
│       │       └── index.tsx
│       ├── features
│       │   ├── auth
│       │   │   ├── hooks
│       │   │   │   └── useAuth.ts
│       │   │   └── index.ts
│       │   ├── prediction
│       │   │   ├── components
│       │   │   │   └── PredictionPanel.tsx
│       │   │   └── index.ts
│       │   ├── projects
│       │   │   ├── hooks
│       │   │   │   └── useProjects.ts
│       │   │   └── index.ts
│       │   └── upload
│       │       ├── components
│       │       │   └── DatasetUploadWizard.tsx
│       │       └── index.ts
│       ├── lib
│       │   ├── config
│       │   │   └── env.ts
│       │   ├── supabase
│       │   │   └── client.ts
│       │   └── utils
│       │       └── index.ts
│       └── models
│           ├── index.ts
│           ├── prediction.ts
│           ├── project.ts
│           └── user.ts
├── ml-service
│   ├── app
│   │   ├── api
│   │   │   └── routes
│   │   │       ├── health.py
│   │   │       ├── predict.py
│   │   │       └── train.py
│   │   ├── application
│   │   │   └── training
│   │   │       ├── __init__.py
│   │   │       ├── automl_pipeline.py
│   │   │       ├── preprocessor.py
│   │   │       └── trainer.py
│   │   ├── core
│   │   │   ├── config.py
│   │   │   └── supabase_client.py
│   │   ├── domain
│   │   │   ├── __init__.py
│   │   │   └── models.py
│   │   ├── infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── model_cache.py
│   │   │   └── supabase_storage.py
│   │   ├── main.py
│   │   ├── models
│   │   ├── services
│   │   └── utils
│   └── test
│       ├── test_health.py
│       └── test_pipeline.py
├── supabase
│   └── init.sql
└── tree.md