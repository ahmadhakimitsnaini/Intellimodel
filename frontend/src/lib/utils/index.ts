// Shared frontend utilities: classNames, formatting, time helpers.

export function cn(
  ...classes: Array<string | false | null | undefined>
): string {
  return classes.filter(Boolean).join(" ");
}

export function truncate(value: string, maxLength: number): string {
  if (value.length <= maxLength) return value;
  return `${value.slice(0, maxLength - 1)}…`;
}

export function timeAgo(isoDate: string | null | undefined): string {
  if (!isoDate) return "unknown";
  const date = new Date(isoDate);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();

  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return "just now";

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min${minutes === 1 ? "" : "s"} ago`;

  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;

  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} day${days === 1 ? "" : "s"} ago`;

  const weeks = Math.floor(days / 7);
  if (weeks < 4) return `${weeks} week${weeks === 1 ? "" : "s"} ago`;

  const months = Math.floor(days / 30);
  if (months < 12) return `${months} month${months === 1 ? "" : "s"} ago`;

  const years = Math.floor(days / 365);
  return `${years} year${years === 1 ? "" : "s"} ago`;
}

export function formatScore(score: number | null | undefined): string {
  if (score == null || Number.isNaN(score)) return "—";
  return score.toFixed(4);
}

export function formatMetricName(metric: string | null | undefined): string {
  if (!metric) return "score";
  return metric
    .replace(/_/g, " ")
    .replace(/\bf1\b/i, "F1")
    .replace(/\br2\b/i, "R²")
    .replace(/\bmae\b/i, "MAE")
    .replace(/\brmse\b/i, "RMSE")
    .replace(/^\w/, (c) => c.toUpperCase());
}

export function formatModelName(name: string | null | undefined): string {
  if (!name) return "Unknown model";
  // Split CamelCase model names into nicer labels
  return name
    .replace(/Classifier$/i, " Classifier")
    .replace(/Regressor$/i, " Regressor")
    .replace(/([a-z])([A-Z])/g, "$1 $2");
}

