"use client";

/**
 * components/upload/CSVUploader.tsx
 *
 * Full upload flow:
 *  1. Drag-and-drop or click-to-browse CSV selection
 *  2. Client-side Papa.parse preview (first 5 rows + column list)
 *  3. Target column selector with inferred type hints
 *  4. Direct-to-Supabase Storage upload (datasets bucket)
 *  5. Project DB row creation
 *  6. FastAPI /train trigger
 *  7. Redirect to project detail page
 */

import { useState, useCallback, useRef } from "react";
import Papa from "papaparse";
import { supabase, FASTAPI_URL, type Project } from "@/lib/supabase";
import { cn } from "@/lib/utils";
import {
  Upload, FileText, X, ChevronDown, Loader2,
  CheckCircle2, AlertCircle, Table2, Target,
} from "lucide-react";

interface CSVUploaderProps {
  userId: string;
  onSuccess: (project: Project) => void;
}

interface ParsedPreview {
  columns: ColumnMeta[];
  rows: Record<string, string>[];
  totalRows: number;
  fileName: string;
}

interface ColumnMeta {
  name: string;
  sample: string[];
  inferredType: "numeric" | "categorical" | "text";
  uniqueCount: number;
}

type UploadStep = "idle" | "parsed" | "uploading" | "training" | "done" | "error";

function inferColumnType(samples: string[]): ColumnMeta["inferredType"] {
  const nonEmpty = samples.filter(Boolean);
  if (nonEmpty.length === 0) return "text";
  const numericCount = nonEmpty.filter((v) => !isNaN(Number(v))).length;
  if (numericCount / nonEmpty.length > 0.8) return "numeric";
  const uniqueRatio = new Set(nonEmpty).size / nonEmpty.length;
  return uniqueRatio < 0.3 ? "categorical" : "text";
}

export function CSVUploader({ userId, onSuccess }: CSVUploaderProps) {
  const [step, setStep] = useState<UploadStep>("idle");
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ParsedPreview | null>(null);
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [projectName, setProjectName] = useState("");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Parse CSV client-side ─────────────────────────────────────────────────

  const parseFile = useCallback((f: File) => {
    setError(null);
    Papa.parse(f, {
      header: true,
      skipEmptyLines: true,
      preview: 200, // parse up to 200 rows for preview & type inference
      complete(results) {
        if (!results.meta.fields?.length) {
          setError("Could not parse CSV headers. Ensure the first row contains column names.");
          return;
        }
        const rows = results.data as Record<string, string>[];
        const columns: ColumnMeta[] = results.meta.fields.map((col) => {
          const samples = rows.slice(0, 100).map((r) => r[col] ?? "");
          return {
            name: col,
            sample: samples.slice(0, 3),
            inferredType: inferColumnType(samples),
            uniqueCount: new Set(samples.filter(Boolean)).size,
          };
        });
        setPreview({
          columns,
          rows: rows.slice(0, 5),
          totalRows: rows.length,
          fileName: f.name,
        });
        setProjectName(f.name.replace(/\.csv$/i, "").replace(/[_-]/g, " "));
        setStep("parsed");
      },
      error(err) {
        setError(`Parse error: ${err.message}`);
      },
    });
  }, []);

  const handleFileSelect = useCallback(
    (f: File) => {
      if (!f.name.toLowerCase().endsWith(".csv")) {
        setError("Only .csv files are supported.");
        return;
      }
      if (f.size > 50 * 1024 * 1024) {
        setError("File exceeds 50 MB limit.");
        return;
      }
      setFile(f);
      parseFile(f);
    },
    [parseFile]
  );

  // ── Drag handlers ─────────────────────────────────────────────────────────

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFileSelect(f);
  };

  // ── Upload + train trigger ────────────────────────────────────────────────

  const handleLaunch = async () => {
    if (!file || !preview || !targetColumn || !projectName.trim()) return;
    setStep("uploading");
    setError(null);
    setProgress(0);

    try {
      // 1. Create project row in Supabase (pending status)
      const { data: project, error: projectErr } = await supabase
        .from("projects")
        .insert({
          user_id: userId,
          name: projectName.trim(),
          status: "pending",
          dataset_filename: file.name,
          target_column: targetColumn,
        })
        .select()
        .single();
      if (projectErr) throw projectErr;
      const projectId = project.id;

      // 2. Upload CSV directly to Supabase Storage (datasets bucket)
      //    Path: {user_id}/{project_id}/{original_filename}
      const storagePath = `${userId}/${projectId}/${file.name}`;
      setProgress(15);

      const { error: uploadErr } = await supabase.storage
        .from("datasets")
        .upload(storagePath, file, {
          contentType: "text/csv",
          upsert: false,
        });
      if (uploadErr) throw uploadErr;
      setProgress(50);

      // 3. Update project row with dataset_path
      await supabase
        .from("projects")
        .update({ dataset_path: storagePath })
        .eq("id", projectId);
      setProgress(60);

      // 4. Trigger FastAPI training pipeline
      setStep("training");
      setProgress(70);

      const trainRes = await fetch(`${FASTAPI_URL}/train/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_id: projectId,
          file_path: storagePath,
          target_column: targetColumn,
        }),
      });

      if (!trainRes.ok) {
        const body = await trainRes.json().catch(() => ({}));
        throw new Error(body?.detail ?? `Training API error: ${trainRes.status}`);
      }

      setProgress(100);
      setStep("done");

      // Return the full project for navigation
      const { data: updatedProject } = await supabase
        .from("projects")
        .select("*")
        .eq("id", projectId)
        .single();

      onSuccess((updatedProject ?? project) as Project);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setStep("error");
    }
  };

  const reset = () => {
    setStep("idle");
    setFile(null);
    setPreview(null);
    setTargetColumn("");
    setProjectName("");
    setProgress(0);
    setError(null);
  };

  // ── Render ────────────────────────────────────────────────────────────────

  if (step === "done") {
    return (
      <div className="card card-accent p-10 text-center animate-fade-in">
        <div className="w-14 h-14 rounded-full bg-jade-glow border border-jade/20
                        flex items-center justify-center mx-auto mb-4">
          <CheckCircle2 className="w-7 h-7 text-jade" />
        </div>
        <h3 className="font-display text-xl text-ink-1 mb-2">Training Launched</h3>
        <p className="text-ink-3 text-sm font-mono">
          The AutoML pipeline is running. Redirecting to your project…
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-5">

      {/* Drop zone */}
      {step === "idle" && (
        <div
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            "relative border-2 border-dashed rounded-xl p-12",
            "flex flex-col items-center justify-center gap-4",
            "cursor-pointer transition-all duration-200 group",
            dragging
              ? "border-copper bg-copper-glow scale-[1.01]"
              : "border-surface-border-bright hover:border-copper/50 hover:bg-surface-3/40"
          )}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="sr-only"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFileSelect(f); }}
          />

          {/* Animated upload icon */}
          <div className={cn(
            "w-16 h-16 rounded-2xl border flex items-center justify-center",
            "transition-all duration-300",
            dragging
              ? "bg-copper/20 border-copper shadow-copper-glow"
              : "bg-surface-3 border-surface-border group-hover:border-copper/40"
          )}>
            <Upload className={cn(
              "w-8 h-8 transition-all duration-300",
              dragging ? "text-copper" : "text-ink-4 group-hover:text-copper/70"
            )} />
          </div>

          <div className="text-center">
            <p className="font-sans font-medium text-ink-2 mb-1">
              Drop your CSV here
            </p>
            <p className="text-ink-4 text-sm font-mono">
              or click to browse · max 50 MB
            </p>
          </div>

          {dragging && (
            <div className="absolute inset-0 rounded-xl bg-copper/5 pointer-events-none animate-fade-in" />
          )}
        </div>
      )}

      {/* File selected — show preview */}
      {step === "parsed" && preview && (
        <div className="space-y-5 animate-fade-up">

          {/* File info bar */}
          <div className="flex items-center justify-between p-3 bg-surface-2
                          border border-surface-border rounded-lg">
            <div className="flex items-center gap-2.5">
              <FileText className="w-4 h-4 text-copper" />
              <span className="font-mono text-sm text-ink-1">{preview.fileName}</span>
              <span className="badge badge-pending">
                {preview.columns.length} columns
              </span>
              <span className="badge badge-pending">
                ~{preview.totalRows} rows
              </span>
            </div>
            <button onClick={reset} className="btn-ghost p-1.5">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Project name */}
          <div>
            <label className="mono-label block mb-1.5">Project Name</label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="input-base"
              placeholder="My churn prediction model"
            />
          </div>

          {/* Target column selector */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-3.5 h-3.5 text-copper" />
              <label className="mono-label">Target Column <span className="text-copper">*</span></label>
            </div>
            <p className="text-ink-4 text-xs font-mono mb-3">
              The column you want the model to predict.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {preview.columns.map((col) => (
                <button
                  key={col.name}
                  onClick={() => setTargetColumn(col.name)}
                  className={cn(
                    "text-left p-3 rounded-lg border transition-all duration-150",
                    "group relative overflow-hidden",
                    targetColumn === col.name
                      ? "border-copper bg-copper-glow shadow-copper-glow"
                      : "border-surface-border bg-surface-2 hover:border-copper/40 hover:bg-surface-3"
                  )}
                >
                  <div className="flex items-start justify-between gap-1 mb-1.5">
                    <span className={cn(
                      "font-mono text-xs font-medium truncate",
                      targetColumn === col.name ? "text-copper" : "text-ink-1"
                    )}>
                      {col.name}
                    </span>
                    <span className={cn(
                      "shrink-0 text-xs font-mono px-1.5 py-0.5 rounded",
                      col.inferredType === "numeric"
                        ? "bg-jade/10 text-jade"
                        : col.inferredType === "categorical"
                        ? "bg-copper/10 text-copper-light"
                        : "bg-ink-5 text-ink-3"
                    )}>
                      {col.inferredType === "numeric" ? "num" : col.inferredType === "categorical" ? "cat" : "txt"}
                    </span>
                  </div>
                  <p className="text-ink-4 text-xs font-mono truncate">
                    {col.sample.filter(Boolean).join(", ") || "—"}
                  </p>
                  {targetColumn === col.name && (
                    <div className="absolute top-0 left-0 right-0 h-px bg-copper-gradient" />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Data preview table */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Table2 className="w-3.5 h-3.5 text-ink-4" />
              <span className="mono-label">Data Preview (first 5 rows)</span>
            </div>
            <div className="overflow-x-auto rounded-lg border border-surface-border">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-surface-border">
                    {preview.columns.map((col) => (
                      <th
                        key={col.name}
                        className={cn(
                          "px-3 py-2.5 text-left whitespace-nowrap font-medium",
                          "first:pl-4",
                          targetColumn === col.name
                            ? "text-copper bg-copper-glow"
                            : "text-ink-3 bg-surface-2"
                        )}
                      >
                        {targetColumn === col.name && (
                          <Target className="inline w-3 h-3 mr-1 mb-0.5" />
                        )}
                        {col.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.rows.map((row, i) => (
                    <tr
                      key={i}
                      className="border-b border-surface-border/50 hover:bg-surface-3/30 transition-colors"
                    >
                      {preview.columns.map((col) => (
                        <td
                          key={col.name}
                          className={cn(
                            "px-3 py-2 text-ink-3 max-w-[120px] truncate",
                            "first:pl-4",
                            targetColumn === col.name && "text-copper-light bg-copper/5"
                          )}
                        >
                          {row[col.name] ?? <span className="text-ink-5">null</span>}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-md
                            bg-crimson-glow border border-crimson/20 text-crimson">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <p className="text-sm font-mono">{error}</p>
            </div>
          )}

          {/* Launch button */}
          <button
            onClick={handleLaunch}
            disabled={!targetColumn || !projectName.trim()}
            className="btn-primary w-full py-3"
          >
            Launch Training Pipeline
          </button>
        </div>
      )}

      {/* Uploading / training state */}
      {(step === "uploading" || step === "training") && (
        <div className="card card-accent p-8 animate-fade-in">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-10 h-10 rounded-lg bg-copper/10 border border-copper/30
                            flex items-center justify-center shrink-0">
              <Loader2 className="w-5 h-5 text-copper animate-spin" />
            </div>
            <div>
              <h3 className="font-sans font-medium text-ink-1">
                {step === "uploading" ? "Uploading dataset…" : "Queuing training job…"}
              </h3>
              <p className="text-ink-4 text-xs font-mono mt-0.5">
                {step === "uploading"
                  ? "Uploading CSV to Supabase Storage"
                  : "Sending job to AutoML pipeline"}
              </p>
            </div>
          </div>

          {/* Progress bar */}
          <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
            <div
              className="h-full bg-copper-gradient rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="mt-2 text-right font-mono text-xs text-ink-4">{progress}%</p>
        </div>
      )}

      {/* Error state */}
      {step === "error" && (
        <div className="space-y-3 animate-fade-in">
          <div className="flex items-start gap-3 p-4 rounded-lg
                          bg-crimson-glow border border-crimson/20">
            <AlertCircle className="w-5 h-5 text-crimson mt-0.5 shrink-0" />
            <div>
              <p className="font-medium text-crimson text-sm">Upload failed</p>
              <p className="text-crimson/70 text-xs font-mono mt-1">{error}</p>
            </div>
          </div>
          <button onClick={reset} className="btn-secondary w-full">
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}
