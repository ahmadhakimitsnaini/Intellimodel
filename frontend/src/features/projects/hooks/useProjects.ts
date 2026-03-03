/**
 * hooks/useProjects.ts
 *
 * Fetches and subscribes to the user's projects table in real-time.
 *
 * Uses Supabase Realtime to listen for INSERT / UPDATE events so the
 * dashboard updates automatically when the FastAPI worker writes back
 * status="completed" — no manual polling required.
 *
 * Additionally exposes a `pollProject` utility for the training progress
 * view, which falls back to interval-based polling for environments where
 * Realtime channels are unavailable.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { supabase, type Project } from "@/lib/supabase";

interface UseProjectsResult {
  projects: Project[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  createProject: (data: CreateProjectInput) => Promise<Project>;
  deleteProject: (id: string) => Promise<void>;
}

interface CreateProjectInput {
  name: string;
  description?: string;
}

export function useProjects(userId: string | undefined): UseProjectsResult {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProjects = useCallback(async () => {
    if (!userId) return;
    setError(null);
    try {
      const { data, error: err } = await supabase
        .from("projects")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false });

      if (err) throw err;
      setProjects(data ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to load projects");
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    if (!userId) { setLoading(false); return; }

    fetchProjects();

    // Subscribe to realtime changes on the projects table for this user
    const channel = supabase
      .channel(`projects:${userId}`)
      .on(
        "postgres_changes",
        {
          event: "*",        // INSERT, UPDATE, DELETE
          schema: "public",
          table: "projects",
          filter: `user_id=eq.${userId}`,
        },
        (payload) => {
          if (payload.eventType === "INSERT") {
            setProjects((prev) => [payload.new as Project, ...prev]);
          } else if (payload.eventType === "UPDATE") {
            setProjects((prev) =>
              prev.map((p) =>
                p.id === (payload.new as Project).id
                  ? { ...p, ...(payload.new as Project) }
                  : p
              )
            );
          } else if (payload.eventType === "DELETE") {
            setProjects((prev) =>
              prev.filter((p) => p.id !== (payload.old as Project).id)
            );
          }
        }
      )
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [userId, fetchProjects]);

  const createProject = useCallback(
    async (data: CreateProjectInput): Promise<Project> => {
      if (!userId) throw new Error("Not authenticated");
      const { data: created, error: err } = await supabase
        .from("projects")
        .insert({ ...data, user_id: userId, status: "pending" })
        .select()
        .single();
      if (err) throw err;
      return created as Project;
    },
    [userId]
  );

  const deleteProject = useCallback(async (id: string) => {
    const { error: err } = await supabase
      .from("projects")
      .delete()
      .eq("id", id);
    if (err) throw err;
  }, []);

  return {
    projects,
    loading,
    error,
    refetch: fetchProjects,
    createProject,
    deleteProject,
  };
}

// ── Single-project polling hook ───────────────────────────────────────────────
// Used on the training progress page to watch a single project's status.

interface UseSingleProjectResult {
  project: Project | null;
  loading: boolean;
}

export function useSingleProject(projectId: string | undefined): UseSingleProjectResult {
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetch = useCallback(async () => {
    if (!projectId) return;
    const { data } = await supabase
      .from("projects")
      .select("*")
      .eq("id", projectId)
      .single();
    if (data) setProject(data as Project);
    setLoading(false);
  }, [projectId]);

  useEffect(() => {
    if (!projectId) return;
    fetch();

    // Realtime subscription
    const channel = supabase
      .channel(`project:${projectId}`)
      .on(
        "postgres_changes",
        {
          event: "UPDATE",
          schema: "public",
          table: "projects",
          filter: `id=eq.${projectId}`,
        },
        (payload) => {
          setProject((prev) =>
            prev ? { ...prev, ...(payload.new as Project) } : (payload.new as Project)
          );
        }
      )
      .subscribe();

    // Interval fallback (polls every 3s while status is training/pending)
    intervalRef.current = setInterval(async () => {
      const { data } = await supabase
        .from("projects")
        .select("status, winning_model, accuracy_score, metric_name, all_scores, task_type, trained_at, error_message")
        .eq("id", projectId)
        .single();
      if (data) {
        setProject((prev) => prev ? { ...prev, ...data } : null);
        // Stop polling once terminal state reached
        if (data.status === "completed" || data.status === "failed") {
          if (intervalRef.current) clearInterval(intervalRef.current);
        }
      }
    }, 3000);

    return () => {
      supabase.removeChannel(channel);
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [projectId, fetch]);

  return { project, loading };
}
