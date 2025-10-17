import { ChevronLeft, RefreshCw } from "lucide-react";

import { Button } from "./ui/button";

interface ChatHeaderProps {
  partnerName: string;
  unreadCount?: number;
  onBack?: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
}

const ChatHeader = ({ partnerName, unreadCount = 0, onBack, onRefresh, isRefreshing }: ChatHeaderProps) => {
  return (
    <div className="flex items-center justify-between border-b border-gray-200 bg-white/95 px-4 py-3 shadow-sm backdrop-blur">
      <div className="flex items-center gap-3">
        {onBack ? (
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="md:hidden"
            onClick={onBack}
            aria-label="戻る"
          >
            <ChevronLeft className="h-5 w-5" />
          </Button>
        ) : null}
        <div>
          <p className="text-base font-semibold text-gray-900">{partnerName}</p>
          {unreadCount > 0 ? (
            <span className="mt-1 inline-flex items-center rounded-full bg-blue-500 px-2 py-0.5 text-[11px] font-semibold text-white shadow-sm">
              未読 {unreadCount}
            </span>
          ) : null}
        </div>
      </div>
      {onRefresh ? (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onRefresh}
          disabled={isRefreshing}
          aria-label="更新"
        >
          <RefreshCw className="h-4 w-4" />
        </Button>
      ) : null}
    </div>
  );
};

export default ChatHeader;
