"use client";

export function TrainingPulse({ message = "Training in progress…" }: { message?: string }) {
  return (
    <div className="flex items-center gap-3 p-4 rounded-lg bg-copper-glow border border-copper/20">
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