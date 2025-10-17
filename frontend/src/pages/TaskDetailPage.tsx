import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { CheckCircle2, Download, Loader2 } from "lucide-react";

import DeadlineCountdown from "../components/DeadlineCountdown";
import FileUploader from "../components/FileUploader";
import SubmissionList from "../components/SubmissionList";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";
import {
  API_BASE_URL,
  fetchTaskDetail,
  fetchTaskSubmissions,
  submitTask,
  type TaskDetail,
  type TaskSubmission,
} from "../services/api";

const TaskDetailPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const params = useParams<{ taskId: string }>();
  const taskId = params.taskId ? Number.parseInt(params.taskId, 10) : Number.NaN;

  const [task, setTask] = useState<TaskDetail | null>(null);
  const [submissions, setSubmissions] = useState<TaskSubmission[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  const loadData = useCallback(async () => {
    if (Number.isNaN(taskId)) {
      setError("課題IDが不正です");
      setIsLoading(false);
      return;
    }
    try {
      setIsLoading(true);
      const detail = await fetchTaskDetail(taskId);
      setTask(detail);
      setError(null);
      if (user?.role === "teacher") {
        const submissionList = await fetchTaskSubmissions(taskId);
        setSubmissions(submissionList);
      }
    } catch (requestError) {
      console.error(requestError);
      setError("課題情報の取得に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  }, [taskId, user?.role]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const handleSubmission = useCallback(async () => {
    if (!task || !file) {
      return;
    }
    setIsSubmitting(true);
    setShowSuccess(false);
    setError(null);
    try {
      await submitTask(task.id, file);
      setFile(null);
      setShowSuccess(true);
      const updated = await fetchTaskDetail(task.id);
      setTask(updated);
    } catch (requestError) {
      console.error(requestError);
      setError("提出に失敗しました。");
    } finally {
      setIsSubmitting(false);
    }
  }, [file, task]);

  const refreshSubmissions = useCallback(async () => {
    if (!task || user?.role !== "teacher") {
      return;
    }
    try {
      const submissionList = await fetchTaskSubmissions(task.id);
      setSubmissions(submissionList);
      setError(null);
    } catch (requestError) {
      console.error(requestError);
    }
  }, [task, user?.role]);

  const statusBadge = useMemo(() => {
    if (!task || user?.role !== "student") {
      return null;
    }
    const status = task.student_status ?? "pending";
    const map = {
      submitted: { text: "提出済み", className: "bg-green-100 text-green-700" },
      late: { text: "遅れて提出", className: "bg-yellow-100 text-yellow-700" },
      overdue: { text: "締切超過", className: "bg-red-100 text-red-600" },
      pending: { text: "未提出", className: "bg-blue-100 text-blue-600" },
    } as const;
    return map[status] ?? map.pending;
  }, [task, user?.role]);

  if (!user) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="flex h-96 items-center justify-center rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
        読み込み中...
      </div>
    );
  }

  if (!task) {
    return (
      <div className="flex h-96 flex-col items-center justify-center gap-4 rounded-3xl border border-dashed border-gray-200 bg-white text-sm text-gray-500">
        {error ?? "課題が見つかりませんでした。"}
        <Button variant="outline" onClick={() => navigate("/tasks")}>一覧へ戻る</Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">{task.title}</h1>
          <p className="mt-1 text-sm text-gray-500">担当教員: {task.teacher_name ?? "-"}</p>
        </div>
        <Button variant="outline" onClick={() => navigate("/tasks")}>一覧へ戻る</Button>
      </div>

      {error ? (
        <div className="rounded-2xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-6">
          <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm">
            <div className="flex flex-wrap items-center gap-3">
              <DeadlineCountdown deadline={task.deadline} />
              {statusBadge ? (
                <span className={`rounded-full px-3 py-1 text-xs font-semibold ${statusBadge.className}`}>
                  {statusBadge.text}
                </span>
              ) : null}
            </div>
            <p className="mt-4 whitespace-pre-wrap text-sm leading-relaxed text-gray-700">
              {task.description || "詳細はありません。"}
            </p>
            {task.file_url ? (
              <a
                href={`${API_BASE_URL}${task.file_url}`}
                target="_blank"
                rel="noreferrer"
                className="mt-6 inline-flex items-center gap-2 rounded-full bg-blue-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-600"
              >
                <Download className="h-4 w-4" />
                {task.original_filename ?? "課題ファイルを開く"}
              </a>
            ) : null}
            {user.role === "teacher" ? (
              <div className="mt-6 space-y-2 text-sm text-gray-600">
                <p className="font-semibold text-gray-800">配布先</p>
                <ul className="flex flex-wrap gap-2">
                  {task.students.map((student) => (
                    <li key={student.id} className="rounded-full bg-gray-100 px-3 py-1 text-xs font-semibold text-gray-600">
                      {student.username}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>

          {user.role === "teacher" ? (
            <SubmissionList submissions={submissions} onRefresh={refreshSubmissions} />
          ) : (
            <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-sm">
              <h2 className="text-base font-semibold text-gray-800">提出</h2>
              <p className="mt-1 text-sm text-gray-500">ファイルを添付して提出ボタンを押してください。</p>
              <div className="mt-4 space-y-4">
                <FileUploader file={file} onChange={setFile} label="提出ファイルを選択" />
                <Button
                  type="button"
                  onClick={() => void handleSubmission()}
                  disabled={!file || isSubmitting}
                  className="flex items-center gap-2"
                >
                  {isSubmitting ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                  {isSubmitting ? "送信中" : "提出する"}
                </Button>
                {showSuccess ? (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-2 text-sm font-semibold text-green-600"
                  >
                    <CheckCircle2 className="h-5 w-5" /> 提出が完了しました！
                  </motion.div>
                ) : null}
                {task.submitted_at ? (
                  <p className="text-xs text-gray-500">
                    最終提出: {new Date(task.submitted_at).toLocaleString("ja-JP", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </p>
                ) : null}
              </div>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="rounded-3xl border border-green-100 bg-green-50 px-5 py-6 text-sm text-green-700 shadow-sm">
            <p className="font-semibold">提出のコツ</p>
            <ul className="mt-3 space-y-2 text-xs leading-relaxed">
              <li>・ 締切前に一度アップロードしておくと安心です。</li>
              <li>・ 課題ファイルは再提出すると最新のものに更新されます。</li>
              <li>・ 遅れた場合は自動的に「遅れて提出」と表示されます。</li>
            </ul>
          </div>
          {task.deadline ? (
            <div className="rounded-3xl border border-blue-100 bg-blue-50 px-5 py-6 text-sm text-blue-700 shadow-sm">
              <p className="font-semibold">締切情報</p>
              <p className="mt-2 text-sm font-medium text-blue-900">
                {new Date(task.deadline).toLocaleString("ja-JP", { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default TaskDetailPage;
