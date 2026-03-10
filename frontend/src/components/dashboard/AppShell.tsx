"use client";

/**
 * components/dashboard/AppShell.tsx
 *
 * Top-level layout wrapper rendered on every authenticated page.
 * Contains the nav bar, user menu, and page content slot.
 */

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useAuth } from "@/features/auth";
import {
  LayoutDashboard, Upload, LogOut, ChevronDown,
  Activity, Settings, User, Cpu,
} from "lucide-react";

/**
 * Properti untuk komponen AppShell.
 * @property {React.ReactNode} children - Konten spesifik halaman yang akan dirender di dalam area main.
 */
interface AppShellProps {
  children: React.ReactNode;
}

// Konfigurasi daftar menu navigasi utama untuk memudahkan penambahan menu di masa depan.
const NAV_ITEMS = [
  { href: "/dashboard",      label: "Dashboard", icon: LayoutDashboard },
  { href: "/dashboard/new",  label: "New Model", icon: Upload },
];

/**
 * AppShell Component
 * * Berfungsi sebagai kerangka global untuk halaman yang terautentikasi (Dashboard).
 * Komponen ini menyertakan Header (Navigasi & User Dropdown), Area Konten Utama, dan Footer.
 * * @param {AppShellProps} props - Props komponen yang membungkus konten dinamis (children).
 * @returns {JSX.Element} Elemen layout dashboard utuh.
 */
export function AppShell({ children }: AppShellProps) {
  // Mengambil data user yang sedang login dan fungsi signOut dari custom hook useAuth
  const { user, signOut } = useAuth();
  
  // Mengambil URL path saat ini untuk menentukan status "aktif" pada menu navigasi
  const pathname = usePathname();
  
  // State untuk mengontrol apakah menu dropdown profil pengguna sedang terbuka atau tertutup
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  // Logika pembentukan inisial pengguna (maksimal 2 huruf kapital):
  // 1. Cek apakah ada full_name, jika ya ambil huruf pertama dari maksimal 2 kata pertama.
  // 2. Jika tidak ada full_name, ambil 2 huruf pertama dari alamat email.
  // 3. Fallback ke "??" jika datanya benar-benar kosong.
  const initials = user?.user_metadata?.full_name
    ? user.user_metadata.full_name.split(" ").map((n: string) => n[0]).join("").slice(0, 2).toUpperCase()
    : user?.email?.slice(0, 2).toUpperCase() ?? "??";

  return (
    <div className="min-h-dvh flex flex-col bg-surface">

      {/* ── Nav bar ────────────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-surface-border
                         bg-surface-1/80 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">

          {/* Brand Logo & Tittle */}
          <Link href="/dashboard" className="flex items-center gap-2.5 group">
            <div className="w-7 h-7 rounded-md bg-copper/10 border border-copper/25
                            flex items-center justify-center
                            group-hover:border-copper/50 transition-colors">
              <span className="font-mono font-bold text-xs text-copper">ML</span>
            </div>
            <span className="font-display text-lg text-ink-1 hidden sm:block">AutoML</span>
          </Link>

          {/* Nav links (Daftar Menu Tengah) */}
          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
              // Validasi menu aktif: Aktif jika path persis sama, atau jika berada di sub-path (kecuali path /dashboard utama)
              const active = pathname === href || (href !== "/dashboard" && pathname.startsWith(href));
              
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-1.5 rounded-md",
                    "font-sans text-sm transition-all duration-150",
                    active
                      ? "bg-surface-3 text-ink-1"
                      : "text-ink-3 hover:text-ink-1 hover:bg-surface-3/50"
                  )}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
          </nav>

          {/* User menu (Dropdown Profil Kanan) */}
          <div className="relative">
            <button
              onClick={() => setUserMenuOpen((v) => !v)}
              className={cn(
                "flex items-center gap-2 px-2 py-1.5 rounded-md",
                "text-ink-3 hover:text-ink-1 hover:bg-surface-3",
                "transition-all duration-150 focus:outline-none"
              )}
            >
              {/* Avatar Inisial */}
              <div className="w-6 h-6 rounded bg-copper/20 border border-copper/30
                              flex items-center justify-center">
                <span className="font-mono text-[10px] font-bold text-copper">{initials}</span>
              </div>
              <span className="font-mono text-xs hidden sm:block max-w-[120px] truncate">
                {user?.email ?? ""}
              </span>
              <ChevronDown className={cn(
                "w-3.5 h-3.5 transition-transform duration-150",
                userMenuOpen && "rotate-180" // Putar ikon panah ke atas saat menu terbuka
              )} />
            </button>

            {/* Dropdown Content */}
            {userMenuOpen && (
              <>
                {/* Overlay layar penuh transparan (Click-away listener): Menutup dropdown jika user klik area di luar menu */}
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setUserMenuOpen(false)}
                />
                <div className="absolute right-0 top-full mt-1.5 w-52 z-20
                                card-elevated rounded-lg py-1 animate-fade-in">
                  
                  {/* Informasi Nama & Email Detail */}
                  <div className="px-3 py-2 border-b border-surface-border">
                    <p className="font-sans text-xs font-medium text-ink-1 truncate">
                      {user?.user_metadata?.full_name ?? "User"}
                    </p>
                    <p className="font-mono text-[11px] text-ink-4 truncate mt-0.5">
                      {user?.email}
                    </p>
                  </div>
                  
                  {/* Tombol Logout */}
                  <div className="py-1">
                    <button
                      className="w-full text-left px-3 py-2 flex items-center gap-2.5
                                 text-ink-3 hover:text-crimson hover:bg-crimson-glow
                                 font-mono text-xs transition-all duration-100"
                      onClick={signOut}
                    >
                      <LogOut className="w-3.5 h-3.5" />
                      Sign out
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      {/* ── Page content ───────────────────────────────────────────────────── */}
      {/* Area pembungkus dinamis di mana komponen halaman (children) akan disisipkan */}
      <main className="flex-1 max-w-6xl mx-auto w-full px-4 sm:px-6 py-8">
        {children}
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="border-t border-surface-border py-4">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 flex items-center justify-between">
          <div className="flex items-center gap-2 text-ink-5">
            <Cpu className="w-3 h-3" />
            <span className="font-mono text-xs">AutoML SaaS · FastAPI + Supabase</span>
          </div>
          <div className="flex items-center gap-1">
            {/* Animasi titik nyala (pulse) sebagai indikator sistem berjalan normal */}
            <div className="w-1.5 h-1.5 rounded-full bg-jade animate-pulse" />
            <span className="font-mono text-xs text-ink-5">All systems operational</span>
          </div>
        </div>
      </footer>
    </div>
  );
}