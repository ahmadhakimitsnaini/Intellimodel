"use client";

import { Database } from "lucide-react";

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