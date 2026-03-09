"use client";

import { Clock, Zap, CheckCircle2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProjectStatus } from "@/models";

const STATUS_CONFIG: Record<ProjectStatus, { label: string; icon: React.ReactNode; className: string }> = { ... };

export function StatusBadge({ status }: { status: ProjectStatus }) {
  const config = STATUS_CONFIG[status];
  return (
    <span className={cn("badge", config.className)}>
      {config.icon}
      {config.label}
    </span>
  );
}