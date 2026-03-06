"use client";

import { Clock, Zap, CheckCircle2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProjectStatus } from "@/models";

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