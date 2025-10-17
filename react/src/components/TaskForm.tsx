import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";

import FileUploader from "./FileUploader";
import { Button } from "./ui/button";
import type { CreateTaskPayload, TaskStudent } from "../services/api";

interface TaskFormProps {
  students: TaskStudent[];
  onSubmit: (payload: CreateTaskPayload) => Promise<void>;
  isSubmitting: boolean;
}

const TaskForm = ({ students, onSubmit, isSubmitting }: TaskFormProps) => {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [deadline, setDeadline] = useState("");
  const [selectedStudents, setSelectedStudents] = useState<number[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);

  const handleToggleStudent = (id: number) => {
    setSelectedStudents((prev) => (prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]));
  };

  const selectedLabel = useMemo(() => {
    if (selectedStudents.length === 0) {
      return "未選択";
    }
    if (selectedStudents.length === students.length) {
      return "全員";
    }
    return `${selectedStudents.length}名を選択`; 
  }, [selectedStudents.length, students.length]);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setShowSuccess(false);
    if (!title.trim()) {
      setError("タイトルを入力してください");
      return;
    }
    if (selectedStudents.length === 0) {
      setError("配布する生徒を1名以上選択してください");
      return;
    }

    const payload: CreateTaskPayload = {
      title: title.trim(),
      description: description.trim() || undefined,
      deadline: deadline ? new Date(deadline).toISOString() : null,
      targetStudents: selectedStudents,
      file,
    };

    try {
      await onSubmit(payload);
      setShowSuccess(true);
    } catch (submitError) {
      console.error(submitError);
      setError("課題の作成に失敗しました。内容を確認してください。");
      setShowSuccess(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>
      ) : null}
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-2">
          <label className="text-sm font-semibold text-gray-700" htmlFor="task-title">
            タイトル
          </label>
          <input
            id="task-title"
            type="text"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            className="w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-900 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
            placeholder="例：50m走の記録分析レポート"
          />
        </div>
        <div className="space-y-2">
          <label className="text-sm font-semibold text-gray-700" htmlFor="task-deadline">
            締切 (任意)
          </label>
          <input
            id="task-deadline"
            type="datetime-local"
            value={deadline}
            onChange={(event) => setDeadline(event.target.value)}
            className="w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm text-gray-900 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
          />
        </div>
      </div>
      <div className="space-y-2">
        <label className="text-sm font-semibold text-gray-700" htmlFor="task-description">
          課題の詳細
        </label>
        <textarea
          id="task-description"
          rows={5}
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          placeholder="レポートの書き方や提出フォーマットの指定があれば記入してください。"
          className="w-full rounded-2xl border border-gray-200 bg-white px-4 py-3 text-sm leading-relaxed text-gray-900 shadow-sm focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-100"
        />
      </div>

      <div className="space-y-3">
        <p className="text-sm font-semibold text-gray-700">配布先の生徒 ({selectedLabel})</p>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {students.map((student) => {
            const checked = selectedStudents.includes(student.id);
            return (
              <label
                key={student.id}
                className={`flex cursor-pointer items-center justify-between rounded-2xl border px-4 py-3 text-sm shadow-sm transition ${
                  checked ? "border-blue-300 bg-blue-50 text-blue-700" : "border-gray-200 bg-white text-gray-700 hover:border-blue-200"
                }`}
              >
                <span className="font-medium">{student.username}</span>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => handleToggleStudent(student.id)}
                  className="h-4 w-4"
                />
              </label>
            );
          })}
        </div>
      </div>

      <div>
        <p className="text-sm font-semibold text-gray-700">課題ファイル (任意)</p>
        <FileUploader file={file} onChange={setFile} label="教材ファイルを添付" />
      </div>

      <div className="flex items-center gap-3">
        <Button type="submit" size="lg" disabled={isSubmitting}>
          {isSubmitting ? "作成中..." : "課題を作成"}
        </Button>
        {showSuccess ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2 text-sm font-semibold text-green-600"
          >
            <CheckCircle2 className="h-5 w-5" />
            作成しました！
          </motion.div>
        ) : null}
      </div>
    </form>
  );
};

export default TaskForm;
