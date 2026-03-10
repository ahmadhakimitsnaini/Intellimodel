"use client";

/**
 * app/dashboard/project/[id]/page.tsx
 *
 * Halaman utama yang menampilkan keseluruhan detail dari satu project Machine Learning.
 * Halaman ini menangani routing dinamis dan me-render UI berdasarkan 3 state utama:
 * * 1. TRAINING STATE  → Menampilkan indikator progress animasi dan polling status.
 * 2. COMPLETED STATE → Menampilkan visualisasi skor (ScoreRing), perbandingan model, 
 * daftar fitur, dan panel prediksi interaktif.
 * 3. FAILED STATE    → Menampilkan pesan error dari backend beserta CTA untuk re-train.
 *
 * Pembaruan data secara real-time diterima melalui Supabase Realtime dengan 
 * fallback interval 3 detik (diatur di dalam custom hook `useSingleProject`).
 */

import { use } from "react";
import Link from "next/link";
import { useSingleProject } from "@/features/projects";
import {
  StatusBadge, ScoreRing, ModelComparisonBar,
  MetricCard, TrainingPulse, Spinner,
} from "@/components/ui";
import { PredictionPanel } from "@/features/prediction";
import {
  cn, formatModelName, formatMetricName, formatScore,
  timeAgo, truncate,
} from "@/lib/utils";
import {
  ArrowLeft, Brain, Target, Layers, Clock, Calendar,
  ChevronRight, Database, BarChart2, Zap, AlertCircle,
  RefreshCw, List,
} from "lucide-react";

/**
 * Properti untuk ProjectPage.
 * @property {Promise<{ id: string }>} params - Parameter rute dinamis dari Next.js.
 */
interface PageProps {
  params: Promise<{ id: string }>;
}

export default function ProjectPage({ params }: PageProps) {
  // Unwrapping params menggunakan hooks `use` (pola standar Next.js 15+ untuk async route params)
  const { id } = use(params);
  
  // Mengambil data project dan status loading dari custom hook yang terhubung ke Supabase
  const { project, loading } = useSingleProject(id);

  // ── 1. Handling State: Loading & Not Found ────────────────────────────────
  
  // Jika data masih di-fetch dari server
  if (loading && !project) {
    return (
      <div className="flex items-center justify-center py-32">
        <Spinner className="w-8 h-8" />
      </div>
    );
  }

  // Jika loading selesai tapi project tidak ditemukan di database
  if (!project) {
    return (
      <div className="page-enter text-center py-24">
        <p className="font-mono text-ink-4 text-sm mb-4">Project not found.</p>
        <Link href="/dashboard" className="btn-secondary">
          <ArrowLeft className="w-4 h-4" /> Back to dashboard
        </Link>
      </div>
    );
  }

  // ── 2. Identifikasi Status Project ────────────────────────────────────────
  const isTraining  = project.status === "training"  || project.status === "pending";
  const isCompleted = project.status === "completed";
  const isFailed    = project.status === "failed";

  return (
    <div className="page-enter space-y-8">

      {/* ── Header Global & Breadcrumb ────────────────────────────────────── */}
      {/* Bagian ini selalu di-render selama project ada, terlepas dari statusnya */}
      <div>
        <div className="flex items-center gap-1.5 text-ink-4 font-mono text-xs mb-4">
          <Link href="/dashboard" className="hover:text-ink-2 transition-colors">
            Dashboard
          </Link>
          <ChevronRight className="w-3 h-3" />
          <span className="text-ink-2 truncate max-w-[200px]">{project.name}</span>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="font-display text-3xl text-ink-1">{project.name}</h1>
              <StatusBadge status={project.status} />
            </div>
            {/* Metadata Project (Tipe Task, Target Kolom, Nama Dataset, Waktu Dibuat) */}
            <div className="flex items-center gap-4 flex-wrap">
              {project.task_type && (
                <span className="flex items-center gap-1.5 font-mono text-xs text-ink-4">
                  <Brain className="w-3.5 h-3.5" />
                  {project.task_type}
                </span>
              )}
              {project.target_column && (
                <span className="flex items-center gap-1.5 font-mono text-xs text-ink-4">
                  <Target className="w-3.5 h-3.5" />
                  target: <span className="text-copper-light">{project.target_column}</span>
                </span>
              )}
              {project.dataset_filename && (
                <span className="flex items-center gap-1.5 font-mono text-xs text-ink-4">
                  <Database className="w-3.5 h-3.5" />
                  {truncate(project.dataset_filename, 30)}
                </span>
              )}
              <span className="flex items-center gap-1.5 font-mono text-xs text-ink-4">
                <Calendar className="w-3.5 h-3.5" />
                {timeAgo(project.created_at)}
              </span>
            </div>
          </div>

          <Link href="/dashboard" className="btn-secondary shrink-0 self-start">
            <ArrowLeft className="w-4 h-4" />
            <span>Dashboard</span>
          </Link>
        </div>
      </div>

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* 3A. TRAINING STATE                                                     */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {isTraining && (
        <div className="space-y-6">
          <TrainingPulse message="AutoML pipeline is running…" />

          {/* Menampilkan rincian tahapan pipeline AutoML yang sedang berjalan */}
          <div className="card p-6">
            <h2 className="font-sans font-medium text-ink-1 mb-5 flex items-center gap-2">
              <Zap className="w-4 h-4 text-copper" />
              Pipeline Stages
            </h2>
            <div className="space-y-4">
              {PIPELINE_STAGES.map(({ label, desc, done }, i) => (
                <div key={i} className="flex items-start gap-3">
                  <div className={cn(
                    "w-6 h-6 rounded-full shrink-0 flex items-center justify-center mt-0.5",
                    "font-mono text-xs font-bold",
                    done
                      ? "bg-jade/20 border border-jade/30 text-jade" // Tahap selesai (Hijau)
                      : i === PIPELINE_STAGES.filter(s => s.done).length
                      ? "bg-copper/20 border border-copper/30 text-copper animate-pulse" // Tahap aktif (Tembaga berkedip)
                      : "bg-surface-3 border border-surface-border text-ink-5" // Tahap menunggu (Abu-abu)
                  )}>
                    {done ? "✓" : i + 1}
                  </div>
                  <div>
                    <p className={cn(
                      "font-mono text-sm font-medium",
                      done ? "text-ink-2" : "text-ink-1"
                    )}>
                      {label}
                    </p>
                    <p className="font-mono text-xs text-ink-4 mt-0.5">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Indikator Live Polling */}
          <div className="flex items-center gap-2 text-ink-4 font-mono text-xs">
            <RefreshCw className="w-3 h-3 animate-spin-slow" />
            Checking for updates every 3 seconds via Supabase Realtime
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* 3B. FAILED STATE                                                       */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {isFailed && (
        <div className="space-y-5">
          <div className="card p-6 border-crimson/20 bg-crimson-glow">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-crimson mt-0.5 shrink-0" />
              <div>
                <h3 className="font-sans font-medium text-crimson mb-2">Training Failed</h3>
                {/* Render pesan error aktual dari backend jika tersedia */}
                {project.error_message && (
                  <pre className="font-mono text-xs text-crimson/80 whitespace-pre-wrap
                                  bg-crimson/5 rounded p-3 mt-2 border border-crimson/10">
                    {project.error_message}
                  </pre>
                )}
              </div>
            </div>
          </div>
          {/* Tombol aksi untuk retry/membuat project baru */}
          <div className="flex gap-3">
            <Link href="/dashboard/new" className="btn-primary">
              <RefreshCw className="w-4 h-4" /> Try Again with New Dataset
            </Link>
            <Link href="/dashboard" className="btn-secondary">
              Back to Dashboard
            </Link>
          </div>
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════════ */}
      {/* 3C. COMPLETED STATE                                                    */}
      {/* ════════════════════════════════════════════════════════════════════ */}
      {isCompleted && (
        <div className="space-y-8">

          {/* Baris KPI (Key Performance Indicators) */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">

            {/* Visualisasi skor melingkar (ScoreRing) yang prominen */}
            <div className="card card-accent p-5 flex flex-col items-center justify-center
                            col-span-2 sm:col-span-1">
              <ScoreRing
                score={project.accuracy_score}
                metricName={project.metric_name}
                size={110}
                strokeWidth={7}
              />
            </div>

            {/* Kartu metrik detail pendukung */}
            <MetricCard
              label="Best Model"
              value={formatModelName(project.winning_model)}
              icon={<Brain className="w-4 h-4" />}
              accent="copper"
            />
            <MetricCard
              label="Task Type"
              value={project.task_type === "classification" ? "Classification" : "Regression"}
              sub={`Target: ${project.target_column}`}
              icon={<Target className="w-4 h-4" />}
              accent="neutral"
            />
            <MetricCard
              label="Features"
              value={project.feature_columns?.length ?? 0}
              sub="input columns"
              icon={<Layers className="w-4 h-4" />}
              accent="neutral"
            />
          </div>

          {/* Grid visualisasi perbandingan model dan daftar fitur dataset */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Bar chart perbandingan skor antar model kandidat */}
            <div className="card card-accent p-6">
              <ModelComparisonBar
                allScores={project.all_scores}
                winnerName={project.winning_model}
                metricName={project.metric_name}
              />
            </div>

            {/* List fitur (kolom input) yang digunakan model untuk inferensi */}
            <div className="card card-accent p-6">
              <div className="flex items-center gap-2 mb-4">
                <List className="w-4 h-4 text-ink-4" />
                <h3 className="font-sans font-medium text-ink-1 text-sm">
                  Feature Columns ({project.feature_columns?.length ?? 0})
                </h3>
              </div>
              <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
                {(project.feature_columns ?? []).map((col) => (
                  <span
                    key={col}
                    className="inline-flex items-center px-2 py-1 rounded
                               bg-surface-3 border border-surface-border
                               font-mono text-xs text-ink-2"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Waktu penyelesaian training */}
          {project.trained_at && (
            <div className="flex items-center gap-2 text-ink-4 font-mono text-xs">
              <Clock className="w-3.5 h-3.5" />
              Training completed {timeAgo(project.trained_at)}
            </div>
          )}

          {/* ── Panel Prediksi (Live Inference) ───────────────────────────── */}
          <div className="card card-accent">
            <div className="p-6 border-b border-surface-border">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-jade-glow border border-jade/20
                                flex items-center justify-center">
                  <BarChart2 className="w-4 h-4 text-jade" />
                </div>
                <div>
                  <h2 className="font-sans font-medium text-ink-1">
                    Run a Prediction
                  </h2>
                  <p className="font-mono text-xs text-ink-4 mt-0.5">
                    Enter feature values to get a live inference from your trained model
                  </p>
                </div>
              </div>
            </div>
            {/* Memanggil komponen PredictionPanel dengan melempar data project */}
            <div className="p-6">
              <PredictionPanel project={project} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Pipeline stage descriptions ───────────────────────────────────────────────
// Penanda "done" saat ini bersifat ilustratif. Implementasi riil nantinya dapat
// melacak sub-status secara dinamis dari tabel database.

const PIPELINE_STAGES = [
  {
    label: "Downloading dataset",
    desc:  "Fetching CSV from Supabase Storage",
    done:  false,
  },
  {
    label: "Preprocessing",
    desc:  "Imputing nulls · encoding categoricals · scaling numerics",
    done:  false,
  },
  {
    label: "Training candidate models",
    desc:  "RandomForest · LogisticRegression · GradientBoosting",
    done:  false,
  },
  {
    label: "Evaluating & selecting winner",
    desc:  "Comparing F1 / R² scores on held-out test set",
    done:  false,
  },
  {
    label: "Serializing & uploading model",
    desc:  "Saving .joblib to Supabase Storage (models bucket)",
    done:  false,
  },
  {
    label: "Updating project record",
    desc:  "Writing results to projects table — triggers Realtime update",
    done:  false,
  },
];