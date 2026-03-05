"use client";

import { Activity } from "lucide-react";

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