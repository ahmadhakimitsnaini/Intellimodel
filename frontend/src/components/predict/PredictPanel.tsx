"use client";

/**
 * components/predict/PredictPanel.tsx
 *
 * Dynamic prediction form built entirely from the project's feature_columns list.
 * Calls GET /predict/{project_id}/schema to understand expected types, then
 * renders one input field per feature. On submit, calls POST /predict/{project_id}.
 *
 * For classification: shows prediction label + probability bar per class.
 * For regression: shows predicted value with context.
 */

import { useState, useEffect } from "react";
import { FASTAPI_URL } from "@/lib/supabase";
import { cn, formatModelName } from "@/lib/utils";
import type { Project } from "@/lib/supabase";
import {
  Loader2, Send, AlertCircle, Sparkles,
  BarChart3, TrendingUp, RefreshCw,
} from "lucide-react";

interface PredictPanelProps {
  project: Project;
}

interface PredictResult {
  prediction: string | number;
  prediction_label: string | null;
  probability: number | null;
  all_probabilities: Record<string, number> | null;
  model_name: string;
  latency_ms: number;
  task_type: string;
}

interface FieldValue {
  value: string;
  error: string | null;
}

export function PredictPanel({ project }: PredictPanelProps) {
  const [fields, setFields] = useState<Record<string, FieldValue>>({});
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const features = project.feature_columns ?? [];
  const taskType = project.task_type;

  // Initialise field state from feature_columns
  useEffect(() => {
    const init: Record<string, FieldValue> = {};
    features.forEach((col) => { init[col] = { value: "", error: null }; });
    setFields(init);
  }, [features.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  const setField = (col: string, value: string) => {
    setFields((prev) => ({ ...prev, [col]: { value, error: null } }));
  };

  const validateAndSubmit = async () => {
    // Validate — all fields required for now
    let hasError = false;
    const updated = { ...fields };
    features.forEach((col) => {
      if (updated[col]?.value.trim() === "") {
        updated[col] = { value: "", error: "Required" };
        hasError = true;
      }
    });
    setFields(updated);
    if (hasError) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Build features object — coerce to number if possible
      const featurePayload: Record<string, string | number> = {};
      features.forEach((col) => {
        const raw = fields[col]?.value.trim() ?? "";
        const num = Number(raw);
        featurePayload[col] = isNaN(num) || raw === "" ? raw : num;
      });

      const res = await fetch(`${FASTAPI_URL}/predict/${project.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: featurePayload }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail ?? `API error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const clearForm = () => {
    const cleared: Record<string, FieldValue> = {};
    features.forEach((col) => { cleared[col] = { value: "", error: null }; });
    setFields(cleared);
    setResult(null);
    setError(null);
  };

  if (features.length === 0) {
    return (
      <div className="p-6 text-center text-ink-4 font-mono text-sm">
        No feature columns available for this project.
      </div>
    );
  }

  return (
    <div className="space-y-6">

      {/* Feature input grid */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <span className="mono-label">Input Features ({features.length})</span>
          {result && (
            <button onClick={clearForm} className="btn-ghost py-1 px-2 text-xs gap-1.5">
              <RefreshCw className="w-3 h-3" /> Clear
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {features.map((col) => {
            const field = fields[col] ?? { value: "", error: null };
            return (
              <div key={col}>
                <label className={cn(
                  "block font-mono text-xs mb-1",
                  field.error ? "text-crimson" : "text-ink-4"
                )}>
                  {col}
                  {field.error && (
                    <span className="ml-1.5 text-crimson">{field.error}</span>
                  )}
                </label>
                <input
                  type="text"
                  value={field.value}
                  onChange={(e) => setField(col, e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") validateAndSubmit();
                  }}
                  placeholder="value"
                  className={cn(
                    "input-base text-xs py-2",
                    field.error && "border-crimson/50 focus:border-crimson focus:ring-crimson/20"
                  )}
                />
              </div>
            );
          })}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={validateAndSubmit}
        disabled={loading}
        className="btn-primary w-full"
      >
        {loading
          ? <><Loader2 className="w-4 h-4 animate-spin" /> Running inference…</>
          : <><Send className="w-4 h-4" /> Get Prediction</>
        }
      </button>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2.5 p-3 rounded-md
                        bg-crimson-glow border border-crimson/20">
          <AlertCircle className="w-4 h-4 text-crimson mt-0.5 shrink-0" />
          <p className="text-sm font-mono text-crimson">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && !error && (
        <div className="card card-accent p-5 animate-fade-up space-y-5">

          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-jade-glow border border-jade/20
                            flex items-center justify-center shrink-0">
              <Sparkles className="w-4 h-4 text-jade" />
            </div>
            <div>
              <p className="font-sans font-medium text-ink-1 text-sm">Prediction Result</p>
              <p className="font-mono text-xs text-ink-4 mt-0.5">
                {formatModelName(result.model_name)} · {result.latency_ms}ms
              </p>
            </div>
          </div>

          {/* Classification result */}
          {taskType === "classification" && result.all_probabilities && (
            <div>
              {/* Winning class */}
              <div className="flex items-center justify-between mb-4 p-3 rounded-lg
                              bg-jade-glow border border-jade/20">
                <div>
                  <p className="mono-label mb-1">Predicted Class</p>
                  <p className="font-mono font-bold text-2xl text-jade">
                    {result.prediction_label ?? String(result.prediction)}
                  </p>
                </div>
                {result.probability != null && (
                  <div className="text-right">
                    <p className="mono-label mb-1">Confidence</p>
                    <p className="font-mono font-bold text-2xl text-jade">
                      {(result.probability * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>

              {/* Probability breakdown */}
              <div>
                <div className="flex items-center gap-1.5 mb-3">
                  <BarChart3 className="w-3.5 h-3.5 text-ink-4" />
                  <span className="mono-label">Class Probabilities</span>
                </div>
                <div className="space-y-2.5">
                  {Object.entries(result.all_probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, prob]) => {
                      const isWinner = cls === String(result.prediction_label ?? result.prediction);
                      return (
                        <div key={cls}>
                          <div className="flex items-center justify-between mb-1">
                            <span className={cn(
                              "font-mono text-xs",
                              isWinner ? "text-jade" : "text-ink-3"
                            )}>
                              {cls}
                            </span>
                            <span className={cn(
                              "font-mono text-xs font-medium",
                              isWinner ? "text-jade" : "text-ink-3"
                            )}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full transition-all duration-500",
                                isWinner ? "bg-jade" : "bg-surface-border-bright"
                              )}
                              style={{ width: `${prob * 100}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            </div>
          )}

          {/* Regression result */}
          {taskType === "regression" && (
            <div className="p-3 rounded-lg bg-copper-glow border border-copper/20">
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp className="w-3.5 h-3.5 text-copper" />
                <p className="mono-label">Predicted Value</p>
              </div>
              <p className="font-mono font-bold text-3xl text-copper">
                {typeof result.prediction === "number"
                  ? result.prediction.toLocaleString(undefined, {
                      maximumFractionDigits: 4,
                    })
                  : String(result.prediction)}
              </p>
              <p className="font-mono text-xs text-copper/60 mt-1">
                target: {project.target_column}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
