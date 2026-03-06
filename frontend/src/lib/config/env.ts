// Centralised access to public environment variables used in the frontend.

export const FASTAPI_URL: string =
  process.env.NEXT_PUBLIC_FASTAPI_URL || "http://localhost:8000";

