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

