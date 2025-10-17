import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import TaskCard from "../components/TaskCard";
import DeadlineCountdown from "../components/DeadlineCountdown";
import Header from "../components/Header";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";
import { fetchTasks, type TaskSummary } from "../services/api";

const TaskListPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      setIsLoading(true);
      const result = await fetchTasks();
      setTasks(result);
      setError(null);
    } catch (requestError) {
      console.error(requestError);
      setError("課題一覧の取得に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadTasks();
  }, [loadTasks]);

  const overdueCount = useMemo(() => {
    if (!user || user.role !== "student") {
      return 0;
    }
    return tasks.filter((task) => task.student_status === "overdue").length;
  }, [tasks, user]);

  const upcomingDeadlines = useMemo(() => {
    return tasks
      .filter((task) => Boolean(task.deadline))
      .sort((a, b) => {
        const aTime = a.deadline ? new Date(a.deadline).getTime() : Number.POSITIVE_INFINITY;
        const bTime = b.deadline ? new Date(b.deadline).getTime() : Number.POSITIVE_INFINITY;
        return aTime - bTime;
      })
      .slice(0, 3);
  }, [tasks]);

  if (!user) {
    return null;
  }

  const role = user.role;

  return (
    <div className="space-y-6">
      <Header
        title="課題"
        subtitle={
          role === "teacher"
            ? "配布した課題の提出状況をリアルタイムで確認できます。"
            : "先生から届いた課題を確認し、締切までに提出しましょう。"
        }
      />

      {error ? (
        <div className="rounded-2xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-4">
        <Button variant="outline" size="sm" onClick={() => void loadTasks()} disabled={isLoading}>
          更新
        </Button>
        {role === "teacher" ? (
          <Button size="sm" onClick={() => navigate("/tasks/create")}>課題を作成</Button>
        ) : null}
        {role === "student" && overdueCount > 0 ? (
          <div className="rounded-full bg-red-100 px-3 py-1 text-xs font-semibold text-red-600">
            未提出の課題 {overdueCount} 件
          </div>
        ) : null}
      </div>

      {isLoading ? (
        <div className="flex h-60 items-center justify-center rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
          読み込み中...
        </div>
      ) : tasks.length === 0 ? (
        <div className="flex h-60 flex-col items-center justify-center rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
          課題はまだありません。
        </div>
      ) : (
        <div className="grid gap-4">
          {tasks.map((task) => (
            <TaskCard
              key={task.id}
              task={task}
              role={role}
              onClick={() => navigate(`/tasks/${task.id}`)}
            />
          ))}
        </div>
      )}

      {role === "teacher" && upcomingDeadlines.length > 0 ? (
        <div className="rounded-3xl border border-blue-100 bg-blue-50 px-6 py-5 text-sm text-blue-700 shadow-sm">
          <p className="font-semibold">直近の締切</p>
          <div className="mt-2 space-y-2">
            {upcomingDeadlines.map((task) => (
              <div key={task.id} className="flex items-center justify-between">
                <span className="font-medium text-blue-900">{task.title}</span>
                <DeadlineCountdown deadline={task.deadline} showIcon={false} />
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default TaskListPage;
