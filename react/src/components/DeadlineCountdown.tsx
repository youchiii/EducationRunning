import { useEffect, useMemo, useState } from "react";
import { Clock } from "lucide-react";

interface DeadlineCountdownProps {
  deadline?: string | null;
  showIcon?: boolean;
}

const getRemaining = (deadline?: string | null) => {
  if (!deadline) {
    return { label: "締切なし", overdue: false };
  }
  const parsed = new Date(deadline);
  if (Number.isNaN(parsed.getTime())) {
    return { label: "締切不明", overdue: false };
  }
  const now = new Date();
  const diff = parsed.getTime() - now.getTime();
  const overdue = diff < 0;
  const absDiff = Math.abs(diff);
  const minutes = Math.floor(absDiff / (1000 * 60));
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (overdue) {
    if (days > 0) {
      return { label: `${days}日超過`, overdue: true };
    }
    if (hours > 0) {
      return { label: `${hours}時間超過`, overdue: true };
    }
    return { label: `${minutes}分超過`, overdue: true };
  }

  if (days > 0) {
    return { label: `${days}日後`, overdue: false };
  }
  if (hours > 0) {
    return { label: `${hours}時間後`, overdue: false };
  }
  return { label: `${minutes}分後`, overdue: false };
};

const DeadlineCountdown = ({ deadline, showIcon = true }: DeadlineCountdownProps) => {
  const [remaining, setRemaining] = useState(() => getRemaining(deadline));

  useEffect(() => {
    setRemaining(getRemaining(deadline));
    const interval = window.setInterval(() => {
      setRemaining(getRemaining(deadline));
    }, 60_000);
    return () => {
      window.clearInterval(interval);
    };
  }, [deadline]);

  const formattedDate = useMemo(() => {
    if (!deadline) {
      return "";
    }
    const parsed = new Date(deadline);
    if (Number.isNaN(parsed.getTime())) {
      return "";
    }
    return parsed.toLocaleString("ja-JP", { year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }, [deadline]);

  const baseClass = remaining.overdue ? "text-red-500" : "text-green-600";

  return (
    <div className={`flex items-center gap-2 text-sm font-medium ${baseClass}`}>
      {showIcon ? <Clock className="h-4 w-4" /> : null}
      <span>{remaining.label}</span>
      {formattedDate ? <span className="text-xs text-gray-500">({formattedDate})</span> : null}
    </div>
  );
};

export default DeadlineCountdown;
