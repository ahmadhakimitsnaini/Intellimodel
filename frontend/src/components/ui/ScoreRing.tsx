"use client";

/**
 * components/ui/ScoreRing.tsx
 *
 * Komponen visualisasi berbentuk cincin (circular progress) untuk menampilkan 
 * skor performa metrik evaluasi model. Dilengkapi dengan animasi pengisian 
 * dan perubahan warna indikatif berdasarkan ambang batas (threshold) skor.
 */

import { formatMetricName } from "@/lib/utils";

/**
 * Properti untuk komponen ScoreRing.
 * @property {number | null} score - Nilai skor metrik (ideal dalam rentang 0.0 hingga 1.0).
 * @property {string | null} [metricName] - Nama metrik yang sedang dievaluasi (contoh: "accuracy", "f1").
 * @property {number} [size=96] - Ukuran diameter keseluruhan komponen dalam piksel.
 * @property {number} [strokeWidth=6] - Ketebalan garis cincin dalam piksel.
 */
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
  // ── 1. Kalkulasi Geometri SVG ─────────────────────────────────────────────
  
  // Radius (r) dihitung dengan mengurangi ketebalan garis dari ukuran total 
  // agar garis SVG tidak terpotong (overflow) di luar area kanvas.
  const r = (size - strokeWidth * 2) / 2;
  
  // Keliling lingkaran (Circumference) = 2 * π * r
  // Nilai ini digunakan sebagai batas maksimal panjang garis (dash) cincin.
  const circumference = 2 * Math.PI * r;
  
  // Memastikan persentase selalu berada di rentang 0 hingga 1.
  // Jika score null, set ke 0.
  const pct = score != null ? Math.max(0, Math.min(1, score)) : 0;
  
  // Menghitung panjang garis yang harus diwarnai sesuai dengan persentase skor.
  const dash = pct * circumference;

  // ── 2. Logika Warna Indikatif ─────────────────────────────────────────────
  
  // Hijau untuk >= 90%, Oranye/Tembaga untuk >= 70%, Merah untuk < 70%.
  // Abu-abu gelap digunakan jika skor belum tersedia (null).
  const color =
    score == null ? "#2a2f3a"
    : pct >= 0.9 ? "#3aad7e"
    : pct >= 0.7 ? "#c97d3a"
    : "#c93a4d";

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      
      {/* Rotasi -90 derajat diterapkan pada SVG agar titik awal (start) dari
        garis lingkaran dimulai dari posisi jam 12 (atas), bukan jam 3 (kanan).
      */}
      <svg width={size} height={size} className="-rotate-90">
        
        {/* Lingkaran Background (Track) */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke="#1c2028"
          strokeWidth={strokeWidth}
        />
        
        {/* Lingkaran Progress (Value) */}
        <circle
          cx={size / 2} cy={size / 2} r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round" // Membuat ujung garis menjadi membulat
          // dasharray memanipulasi garis putus-putus SVG. 
          // Formatnya: "panjang_garis panjang_celah_kosong"
          strokeDasharray={`${dash} ${circumference}`}
          // Animasi transisi yang halus saat nilai skor berubah
          style={{ transition: "stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      
      {/* ── 3. Teks Label di Tengah Cincin ─────────────────────────────────── */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {/* Persentase Skor */}
        <span className="font-mono font-semibold text-ink-1" style={{ fontSize: size * 0.18 }}>
          {score != null ? `${(pct * 100).toFixed(1)}%` : "—"}
        </span>
        {/* Nama Metrik (Opsional) */}
        {metricName && (
          <span className="font-mono text-ink-4 mt-0.5" style={{ fontSize: size * 0.11 }}>
            {formatMetricName(metricName)}
          </span>
        )}
      </div>
    </div>
  );
}