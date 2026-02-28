"use client";

/**
 * components/ui/index.tsx
 *
 * Shared primitive components used across all pages.
 * Keeps the design system consistent and imports DRY.
 */

import { cn, formatScore, formatMetricName, formatModelName } from "@/lib/utils";
import type { ProjectStatus } from "@/lib/supabase";
import {
  Loader2, Clock, Zap, CheckCircle2, XCircle,
  Brain, TrendingUp, Database, Activity,
} from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// StatusBadge
// ─────────────────────────────────────────────────────────────────────────────

const STATUS_CONFIG: Record<
  ProjectStatus,
  { label: string; icon: React.ReactNode; className: string }
> = {
  pending: {
    label: "Pending",
    icon: <Clock className="w-3 h-3" />,
    className: "badge-pending",
  },
  training: {
    label: "Training",
    icon: <Zap className="w-3 h-3 animate-pulse" />,
    className: "badge-training",
  },
  completed: {
    label: "Completed",
    icon: <CheckCircle2 className="w-3 h-3" />,
    className: "badge-completed",
  },
  failed: {
    label: "Failed",
    icon: <XCircle className="w-3 h-3" />,
    className: "badge-failed",
  },
};

export function StatusBadge({ status }: { status: ProjectStatus }) {
  const config = STATUS_CONFIG[status];
  return (
    <span className={cn("badge", config.className)}>
      {config.icon}
      {config.label}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ScoreRing  — circular SVG gauge for accuracy/r² display
// ─────────────────────────────────────────────────────────────────────────────

interface ScoreRingProps {
  score: number | null;
  metricName?: string | null;
  size?: number;
  strokeWidth?: number;
}

export function ScoreRing({
  score,
  metricName,
  size = 96,
  strokeWidth = 6,
}: ScoreRingProps) {
  const r = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * r;
  const pct = score != null ? Math.max(0, Math.min(1, score)) : 0;
  const dash = pct * circumference;

  // Color thresholds
  const color =
    score == null ? "#2a2f3a"
    : pct >= 0.9 ? "#3aad7e"   // jade — great
    : pct >= 0.7 ? "#c97d3a"   // copper — good
    : "#c93a4d";                // crimson — needs work

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        {/* Track */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="#1c2028"
          strokeWidth={strokeWidth}
        />
        {/* Progress */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          style={{ transition: "stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono font-semibold text-ink-1" style={{ fontSize: size * 0.18 }}>
          {score != null ? `${(pct * 100).toFixed(1)}%` : "—"}
        </span>
        {metricName && (
          <span className="font-mono text-ink-4 mt-0.5" style={{ fontSize: size * 0.11 }}>
            {formatMetricName(metricName)}
          </span>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricCard  — single KPI tile
// ─────────────────────────────────────────────────────────────────────────────

interface MetricCardProps {
  label: string;
  value: string | number | null;
  sub?: string;
  icon?: React.ReactNode;
  accent?: "copper" | "jade" | "crimson" | "neutral";
  loading?: boolean;
}

const ACCENT_CLASSES = {
  copper:  "border-copper/20 bg-copper-glow",
  jade:    "border-jade/20 bg-jade-glow",
  crimson: "border-crimson/20 bg-crimson-glow",
  neutral: "border-surface-border bg-surface-2",
};

const ICON_ACCENT = {
  copper:  "text-copper",
  jade:    "text-jade",
  crimson: "text-crimson",
  neutral: "text-ink-4",
};

export function MetricCard({
  label, value, sub, icon, accent = "neutral", loading = false,
}: MetricCardProps) {
  return (
    <div className={cn(
      "rounded-lg border p-4 flex flex-col gap-2",
      ACCENT_CLASSES[accent]
    )}>
      <div className="flex items-center justify-between">
        <span className="mono-label">{label}</span>
        {icon && (
          <span className={cn("opacity-60", ICON_ACCENT[accent])}>{icon}</span>
        )}
      </div>
      {loading ? (
        <div className="skeleton h-6 w-24 rounded" />
      ) : (
        <span className="font-mono font-semibold text-ink-1 text-lg leading-none">
          {value ?? "—"}
        </span>
      )}
      {sub && !loading && (
        <span className="font-mono text-xs text-ink-4">{sub}</span>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelComparisonBar  — horizontal bar chart for all_scores
// ─────────────────────────────────────────────────────────────────────────────

interface ModelComparisonBarProps {
  allScores: Record<string, number> | null;
  winnerName: string | null;
  metricName: string | null;
}

export function ModelComparisonBar({
  allScores, winnerName, metricName,
}: ModelComparisonBarProps) {
  if (!allScores) return null;

  const entries = Object.entries(allScores).sort(([, a], [, b]) => b - a);
  const max = Math.max(...entries.map(([, v]) => v));

  return (
    <div className="space-y-3">
      <span className="mono-label">
        Model Comparison · {formatMetricName(metricName)}
      </span>
      {entries.map(([model, score]) => {
        const isWinner = model === winnerName;
        const pct = max > 0 ? (score / max) * 100 : 0;
        return (
          <div key={model}>
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                {isWinner && <Brain className="w-3.5 h-3.5 text-copper" />}
                <span className={cn(
                  "font-mono text-xs",
                  isWinner ? "text-copper" : "text-ink-3"
                )}>
                  {formatModelName(model)}
                </span>
                {isWinner && (
                  <span className="badge badge-training text-[10px] px-1.5 py-0">winner</span>
                )}
              </div>
              <span className={cn(
                "font-mono text-xs font-medium",
                isWinner ? "text-copper" : "text-ink-3"
              )}>
                {formatScore(score)}
              </span>
            </div>
            <div className="h-1.5 bg-surface-3 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-700",
                  isWinner ? "bg-copper-gradient" : "bg-surface-border-bright"
                )}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainingPulse  — animated indicator for "training" status
// ─────────────────────────────────────────────────────────────────────────────

export function TrainingPulse({ message = "Training in progress…" }: { message?: string }) {
  return (
    <div className="flex items-center gap-3 p-4 rounded-lg
                    bg-copper-glow border border-copper/20">
      <div className="relative shrink-0">
        <div className="w-3 h-3 rounded-full bg-copper animate-pulse" />
        <div className="absolute inset-0 w-3 h-3 rounded-full bg-copper animate-ping opacity-40" />
      </div>
      <div>
        <p className="font-mono text-sm text-copper">{message}</p>
        <p className="font-mono text-xs text-copper/60 mt-0.5">
          Results will appear automatically when training completes
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// EmptyState
// ─────────────────────────────────────────────────────────────────────────────

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-20 px-6 text-center">
      <div className="w-16 h-16 rounded-2xl bg-surface-3 border border-surface-border
                      flex items-center justify-center mb-5 text-ink-4">
        {icon ?? <Database className="w-7 h-7" />}
      </div>
      <h3 className="font-display text-xl text-ink-1 mb-2">{title}</h3>
      <p className="text-ink-4 text-sm font-mono max-w-xs mb-6">{description}</p>
      {action}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Spinner
// ─────────────────────────────────────────────────────────────────────────────

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("animate-spin text-copper", className)} />;
}

// ─────────────────────────────────────────────────────────────────────────────
// PageLoader  — full-screen loader shown during auth check
// ─────────────────────────────────────────────────────────────────────────────

export function PageLoader() {
  return (
    <div className="min-h-dvh flex items-center justify-center bg-surface">
      <div className="flex flex-col items-center gap-4">
        <div className="w-10 h-10 rounded-xl bg-copper/10 border border-copper/20
                        flex items-center justify-center">
          <Activity className="w-5 h-5 text-copper animate-pulse" />
        </div>
        <p className="font-mono text-xs text-ink-4 tracking-widest uppercase">
          Loading…
        </p>
      </div>
    </div>
  );
}
