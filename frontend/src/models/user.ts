export interface UserProfile {
  id: string;
  full_name: string | null;
  avatar_url: string | null;
  plan: "free" | "pro" | "enterprise";
  projects_count: number;
  created_at: string;
}

