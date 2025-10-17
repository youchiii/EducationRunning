import { motion } from "framer-motion";
import { ChevronRight } from "lucide-react";

import DeadlineCountdown from "./DeadlineCountdown";
import type { TaskSummary } from "../services/api";

interface TaskCardProps {
  task: TaskSummary;
  role: "teacher" | "student";
  onClick?: () => void;
}

const statusLabelMap: Record<string, { label: string; className: string }> = {
  submitted: { label: "提出済み", className: "bg-green-100 text-green-700" },
  late: { label: "遅れて提出", className: "bg-yellow-100 text-yellow-700" },
  overdue: { label: "締切超過", className: "bg-red-100 text-red-600" },
  pending: { label: "未提出", className: "bg-blue-100 text-blue-600" },
};

const TaskCard = ({ task, role, onClick }: TaskCardProps) => {
  const studentStatus = task.student_status ?? "pending";
  const displayStatus = statusLabelMap[studentStatus] ?? statusLabelMap.pending;

  const teacherStats = (() => {
    if (role !== "teacher") {
      return null;
    }
    const submitted = task.submitted_count ?? 0;
    const total = task.total_assignees ?? 0;
    const allSubmitted = total > 0 && submitted >= total;
    const deadlineDate = task.deadline ? new Date(task.deadline) : null;
    const deadlineValid = deadlineDate && !Number.isNaN(deadlineDate.getTime());
    const overdue = Boolean(deadlineValid && Date.now() > (deadlineDate?.getTime() ?? 0) && !allSubmitted);
    let tone = "text-blue-600";
    if (allSubmitted) {
      tone = "text-green-600";
    } else if (overdue) {
      tone = "text-red-500";
    }
    return { submitted, total, tone, overdue, allSubmitted };
  })();

  return (
    <motion.button
      type="button"
      onClick={onClick}
      className="w-full"
      whileHover={{ y: -3, boxShadow: "0px 12px 30px rgba(15, 118, 110, 0.14)" }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 260, damping: 20 }}
    >
      <div className="flex w-full items-start justify-between rounded-3xl border border-gray-200 bg-white px-5 py-4 text-left shadow-sm">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-gray-900">{task.title}</h3>
            {role === "student" ? (
              <span className={`rounded-full px-3 py-1 text-xs font-semibold ${displayStatus.className}`}>
                {displayStatus.label}
              </span>
            ) : null}
          </div>
          {task.description ? (
            <p className="max-w-2xl text-sm text-gray-600">{task.description}</p>
          ) : (
            <p className="text-sm text-gray-400">説明はありません。</p>
          )}
          <DeadlineCountdown deadline={task.deadline} />
          {role === "teacher" && teacherStats ? (
            <p className={`text-sm font-semibold ${teacherStats.tone}`}>
              {teacherStats.submitted}/{teacherStats.total ?? 0} 名が提出
            </p>
          ) : null}
        </div>
        <ChevronRight className="mt-1 h-5 w-5 text-gray-300" />
      </div>
    </motion.button>
  );
};

export default TaskCard;
