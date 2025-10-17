import { CornerUpRight } from "lucide-react";

import { cn } from "../lib/utils";
import type { ChatMessage } from "../services/api";

const formatTime = (timestamp: string) => {
  const value = timestamp.replace(" ", "T");
  const date = new Date(value.endsWith("Z") ? value : `${value}`);
  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }
  return date.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
};

interface ChatBubbleProps {
  message: ChatMessage;
  isOwn: boolean;
  replyMessage?: ChatMessage | null;
  onReply?: (message: ChatMessage) => void;
}

const ChatBubble = ({ message, isOwn, replyMessage, onReply }: ChatBubbleProps) => {
  return (
    <div
      className={cn(
        "relative max-w-[78%] rounded-3xl px-4 py-3 shadow-md",
        isOwn ? "bg-green-100 text-gray-900" : "bg-gray-100 text-gray-900",
      )}
    >
      {replyMessage && (
        <div className="mb-2 rounded-2xl border-l-4 border-green-300 bg-white/70 px-3 py-2 text-xs text-gray-600 shadow-inner">
          <p className="text-[11px] font-semibold text-green-600">返信先</p>
          <p className="mt-1 max-h-24 overflow-hidden whitespace-pre-wrap leading-snug">{replyMessage.text}</p>
        </div>
      )}
      <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.text}</p>
      <div className="mt-2 flex items-center justify-between text-[11px] text-gray-500">
        <span>{formatTime(message.timestamp)}</span>
        <div className="flex items-center gap-2">
          {isOwn && (
            <span className={cn("font-medium", message.is_read ? "text-green-600" : "text-gray-400")}>{
              message.is_read ? "既読" : "送信済み"
            }</span>
          )}
          {onReply && (
            <button
              type="button"
              className="flex items-center gap-1 rounded-full px-2 py-1 text-xs font-semibold text-blue-600 transition hover:bg-blue-50"
              onClick={() => onReply(message)}
            >
              <CornerUpRight className="h-4 w-4" />
              返信
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatBubble;
