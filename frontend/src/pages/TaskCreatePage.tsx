import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import TaskForm from "../components/TaskForm";
import Header from "../components/Header";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";
import { createTask, fetchTaskStudents, type CreateTaskPayload, type TaskStudent } from "../services/api";

const TaskCreatePage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [students, setStudents] = useState<TaskStudent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadStudents = useCallback(async () => {
    try {
      setIsLoading(true);
      const list = await fetchTaskStudents();
      setStudents(list);
      setError(null);
    } catch (requestError) {
      console.error(requestError);
      setError("生徒一覧の取得に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (user?.role === "teacher") {
      void loadStudents();
    }
  }, [loadStudents, user?.role]);

  if (!user) {
    return null;
  }

  if (user.role !== "teacher") {
    return (
      <div className="space-y-6">
        <Header title="課題作成" subtitle="課題の作成は教員のみが利用できます。" />
        <div className="flex h-80 flex-col items-center justify-center gap-4 rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
          このページを見る権限がありません。
          <Button variant="outline" onClick={() => navigate("/tasks")}>課題一覧へ戻る</Button>
        </div>
      </div>
    );
  }

  const handleSubmit = async (payload: CreateTaskPayload) => {
    try {
      setIsSubmitting(true);
      setError(null);
      const created = await createTask(payload);
      navigate(`/tasks/${created.id}`);
    } catch (requestError) {
      console.error(requestError);
      setError("課題の作成に失敗しました。");
      throw requestError;
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <Header title="課題作成" subtitle="課題タイトルや締切、生徒を選んで配布しましょう。" />
      {error ? (
        <div className="rounded-2xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      ) : null}
      {isLoading ? (
        <div className="flex h-80 items-center justify-center rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
          読み込み中...
        </div>
      ) : students.length === 0 ? (
        <div className="flex h-80 flex-col items-center justify-center gap-4 rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
          配布先の生徒が見つかりません。生徒を登録してください。
          <Button variant="outline" onClick={() => navigate("/tasks")}>課題一覧へ戻る</Button>
        </div>
      ) : (
        <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm">
          <TaskForm students={students} onSubmit={handleSubmit} isSubmitting={isSubmitting} />
        </div>
      )}
    </div>
  );
};

export default TaskCreatePage;
