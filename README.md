##STRUKTUR DIREKTORI

automl-saas/                  # Folder Utama (Root)
│
├── README.md                 # Dokumentasi arsitektur proyek
│
├── supabase/                 # 🗄️ FOLDER DATABASE
│   └── init.sql              # Skema tabel, fungsi trigger, dan aturan RLS
│
├── ml-service/               # 🧠 FOLDER BACKEND (Python / FastAPI)
│   ├── .env                  # (Buat manual) Isi SUPABASE_URL & SUPABASE_SERVICE_KEY
│   ├── requirements.txt      # Daftar library Python (FastAPI, pandas, scikit-learn, dll)
│   ├── tests/
│   │   ├── test_health.py    # Unit test untuk endpoint health
│   │   └── test_pipeline.py  # Unit test untuk mesin AutoML
│   └── app/
│       ├── main.py           # Entry point aplikasi FastAPI
│       ├── api/
│       │   └── routes/
│       │       ├── health.py # Endpoint /health
│       │       ├── predict.py# Endpoint /predict/{project_id}
│       │       └── train.py  # Endpoint /train
│       ├── core/
│       │   ├── config.py     # Pengaturan environment variables
│       │   └── supabase_client.py # Koneksi Supabase Service Role
│       ├── models/
│       │   └── schemas.py    # Pydantic models (Validasi input/output)
│       ├── services/
│       │   ├── automl.py     # Orkestrator utama (pengatur alur)
│       │   ├── preprocessor.py # Pembersih data otomatis (Data Scientist AI)
│       │   ├── storage.py    # Interaksi baca/tulis ke Supabase Storage & DB
│       │   └── trainer.py    # Pelatih 3 model Machine Learning
│       └── utils/
│           └── model_cache.py# Sistem caching di RAM agar prediksi super cepat
│
└── frontend/                 # 💻 FOLDER FRONTEND (Next.js / React)
    ├── .env.local            # (Buat manual) Isi URL Supabase, Anon Key & URL FastAPI
    ├── package.json          # Konfigurasi dependensi Node.js (React, Tailwind, dll)
    └── src/
        ├── app/
        │   ├── globals.css   # Styling global dan konfigurasi Tailwind CSS
        │   └── dashboard/
        │       └── page.tsx  # Halaman utama (Dashboard) setelah user login
        ├── components/
        │   ├── dashboard/
        │   │   └── AppShell.tsx   # Layout utama (Navbar & Footer)
        │   ├── predict/
        │   │   └── PredictPanel.tsx # Formulir prediksi dinamis
        │   ├── ui/
        │   │   └── index.tsx      # Komponen UI yang bisa dipakai ulang (Badge, Loading, dll)
        │   └── upload/
        │       └── CSVUploader.tsx  # Komponen Drag & Drop file CSV
        ├── hooks/
        │   ├── useAuth.ts         # Hook untuk session login/logout Supabase
        │   └── useProjects.ts     # Hook realtime database untuk daftar project
        └── lib/
            └── supabase.ts        # Setup Supabase client untuk browser (ANON_KEY)