import { X } from "lucide-react";

import type { ChatMessage } from "../services/api";

interface ReplyPreviewProps {
  message: ChatMessage;
  onCancel?: () => void;
}

const ReplyPreview = ({ message, onCancel }: ReplyPreviewProps) => {
  return (
    <div className="mb-3 rounded-2xl border border-green-200 bg-green-50 px-4 py-3 text-xs text-gray-700 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-wide text-green-600">返信先</p>
        {onCancel && (
          <button
            type="button"
            className="rounded-full p-1 text-gray-500 transition hover:bg-white hover:text-gray-700"
            onClick={onCancel}
            aria-label="返信をキャンセル"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm leading-snug text-gray-700">{message.text}</p>
    </div>
  );
};

export default ReplyPreview;
