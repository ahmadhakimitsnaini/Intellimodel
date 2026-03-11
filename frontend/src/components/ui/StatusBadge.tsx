"use client";

/**
 * components/ui/StatusBadge.tsx
 *
 * Komponen UI kecil (badge) untuk menampilkan status proyek saat ini secara visual.
 * Memanfaatkan pemetaan objek (config map) agar render ikon dan warna 
 * lebih dinamis tanpa perlu logika percabangan (if/else) yang rumit.
 */

import { Clock, Zap, CheckCircle2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
// Pastikan tipe ProjectStatus sesuai dengan definisi model Anda (misal: "pending" | "training" | "completed" | "failed")
import type { ProjectStatus } from "@/models";

// ── 1. Konfigurasi Mapping Status ──────────────────────────────────────────
// Mendefinisikan tampilan visual (label teks, ikon, dan kelas CSS warna) untuk setiap status.
const STATUS_CONFIG: Record<ProjectStatus, { label: string; icon: React.ReactNode; className: string }> = {
  pending: { 
    label: "Pending", 
    icon: <Clock className="w-3.5 h-3.5" />, 
    className: "bg-surface-3 text-ink-3 border-surface-border" 
  },
  training: { 
    label: "Training", 
    icon: <Zap className="w-3.5 h-3.5 fill-copper" />, 
    className: "bg-copper/10 text-copper border-copper/20" 
  },
  completed: { 
    label: "Completed", 
    icon: <CheckCircle2 className="w-3.5 h-3.5" />, 
    className: "bg-jade/10 text-jade border-jade/20" 
  },
  failed: { 
    label: "Failed", 
    icon: <XCircle className="w-3.5 h-3.5" />, 
    className: "bg-crimson/10 text-crimson border-crimson/20" 
  },
};

/**
 * Properti untuk komponen StatusBadge.
 * @property {ProjectStatus} status - Status proyek saat ini yang akan menentukan tampilan badge.
 */
interface StatusBadgeProps {
  status: ProjectStatus;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  // ── 2. Logika Render ──────────────────────────────────────────────────────
  // Mengambil konfigurasi spesifik berdasarkan status yang dilempar via props.
  // Fallback (opsional) bisa ditambahkan di sini jika status yang dilempar tidak dikenali.
  const config = STATUS_CONFIG[status];

  // Jika status tidak ada di config (untuk keamanan agar aplikasi tidak crash)
  if (!config) return null;

  return (
    // Menggabungkan class dasar "badge" dengan class spesifik dari config menggunakan utilitas `cn`
    <span className={cn(
      "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border font-mono text-xs font-medium", 
      config.className
    )}>
      {config.icon}
      {config.label}
    </span>
  );
}