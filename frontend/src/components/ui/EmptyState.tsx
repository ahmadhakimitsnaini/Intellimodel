import { EmptyState } from "@/components/ui/EmptyState";
import { FolderPlus } from "lucide-react";

export default function ProjectList() {
  const projects = [];

  if (projects.length === 0) {
    return (
      <EmptyState 
        icon={<FolderPlus className="w-7 h-7" />}
        title="Belum ada proyek"
        description="Anda belum membuat proyek apapun. Silakan buat proyek pertama Anda untuk memulai."
        action={<button className="bg-blue-500 text-white px-4 py-2 rounded">Buat Proyek</button>}
      />
    );
  }

  return (
    // Render daftar proyek di sini
  );
}