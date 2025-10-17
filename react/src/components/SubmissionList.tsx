import { AnimatePresence, motion } from "framer-motion";
import { Download } from "lucide-react";

import { API_BASE_URL } from "../services/api";
import type { TaskSubmission } from "../services/api";
import { Button } from "./ui/button";

interface SubmissionListProps {
  submissions: TaskSubmission[];
  onRefresh?: () => void;
}

const statusBadge: Record<TaskSubmission["status"], { label: string; className: string }> = {
  submitted: { label: "提出済み", className: "bg-green-100 text-green-700" },
  late: { label: "遅れて提出", className: "bg-yellow-100 text-yellow-700" },
  pending: { label: "未提出", className: "bg-blue-100 text-blue-600" },
  overdue: { label: "締切超過", className: "bg-red-100 text-red-600" },
};

const SubmissionList = ({ submissions, onRefresh }: SubmissionListProps) => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-gray-800">提出一覧</h2>
        {onRefresh ? (
          <Button type="button" variant="outline" size="sm" onClick={onRefresh}>
            再読み込み
          </Button>
        ) : null}
      </div>
      <div className="divide-y divide-gray-100 rounded-2xl border border-gray-200 bg-white shadow-sm">
        <div className="grid grid-cols-12 gap-4 px-6 py-3 text-xs font-semibold uppercase tracking-wide text-gray-500">
          <span className="col-span-4">生徒</span>
          <span className="col-span-3">状態</span>
          <span className="col-span-3">提出日時</span>
          <span className="col-span-2">ファイル</span>
        </div>
        <AnimatePresence>
          {submissions.map((item) => {
            const badge = statusBadge[item.status];
            return (
              <motion.div
                key={item.student_id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.2 }}
                className="grid grid-cols-12 gap-4 px-6 py-4 text-sm text-gray-700"
              >
                <span className="col-span-4 font-medium text-gray-900">{item.student_name}</span>
                <span className="col-span-3">
                  <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold ${badge.className}`}>
                    {badge.label}
                  </span>
                </span>
                <span className="col-span-3 text-sm text-gray-500">
                  {item.submitted_at ? new Date(item.submitted_at).toLocaleString("ja-JP", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "-"}
                </span>
                <span className="col-span-2">
                  {item.file_url ? (
                    <a
                      href={`${API_BASE_URL}${item.file_url}`}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 rounded-full bg-blue-500 px-3 py-1 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-600"
                    >
                      <Download className="h-4 w-4" />
                      ダウンロード
                    </a>
                  ) : (
                    <span className="text-xs text-gray-400">なし</span>
                  )}
                </span>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default SubmissionList;
