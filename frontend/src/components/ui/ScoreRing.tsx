"use client";

import { formatMetricName } from "@/lib/utils";

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

  const color =
    score == null ? "#2a2f3a"
    : pct >= 0.9 ? "#3aad7e"
    : pct >= 0.7 ? "#c97d3a"
    : "#c93a4d";

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="#1c2028"
          strokeWidth={strokeWidth}
        />
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