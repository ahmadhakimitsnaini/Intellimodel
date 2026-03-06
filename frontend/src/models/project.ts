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

