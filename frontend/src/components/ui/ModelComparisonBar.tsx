"use client";

import { cn, formatScore, formatMetricName, formatModelName } from "@/lib/utils";
import { Brain } from "lucide-react";

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