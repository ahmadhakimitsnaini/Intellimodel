/**
 * lib/supabase.ts
 *
 * Browser-side Supabase client using the ANON key.
 * This is safe to expose — RLS policies enforce row-level isolation.
 *
 * NEVER import settings that use the SERVICE_ROLE key here.
 * The service role key lives only in the FastAPI environment.
 */

import { createClient } from "@supabase/supabase-js";

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    "Missing Supabase environment variables. " +
      "Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local"
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
  },
});

// ── TypeScript types mirroring the DB schema ──────────────────────────────────

export type ProjectStatus = "pending" | "training" | "completed" | "failed";
export type TaskType = "classification" | "regression";

export interface Project {
  id: string;
  user_id: string;
  name: string;
  description: string | null;
  // Dataset
  dataset_path: string | null;
  dataset_filename: string | null;
  target_column: string | null;
  feature_columns: string[] | null;
  // ML results
  task_type: TaskType | null;
  winning_model: string | null;
  accuracy_score: number | null;
  metric_name: string | null;
  all_scores: Record<string, number> | null;
  model_path: string | null;
  // Status
  status: ProjectStatus;
  error_message: string | null;
  // Timestamps
  created_at: string;
  updated_at: string;
  trained_at: string | null;
}

export interface UserProfile {
  id: string;
  full_name: string | null;
  avatar_url: string | null;
  plan: "free" | "pro" | "enterprise";
  projects_count: number;
  created_at: string;
}

export interface PredictionLog {
  id: string;
  project_id: string;
  user_id: string;
  input_data: Record<string, unknown>;
  prediction: Record<string, unknown>;
  model_version: string | null;
  latency_ms: number | null;
  created_at: string;
}

// ── FastAPI URL ───────────────────────────────────────────────────────────────
export const FASTAPI_URL =
  process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000";
