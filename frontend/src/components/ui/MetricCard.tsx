"use client";

import { cn } from "@/lib/utils";

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