import type { FormEvent, KeyboardEvent } from "react";

import type { ChatMessage } from "../services/api";
import { Button } from "./ui/button";
import ReplyPreview from "./ReplyPreview";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => Promise<void> | void;
  isSending: boolean;
  replyMessage?: ChatMessage | null;
  onCancelReply?: () => void;
}

const ChatInput = ({ value, onChange, onSend, isSending, replyMessage, onCancelReply }: ChatInputProps) => {
  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!value.trim()) {
      return;
    }
    await onSend();
  };

  const handleKeyDown = async (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!value.trim()) {
        return;
      }
      await onSend();
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-gray-200 bg-white px-4 py-4 shadow-inner">
      {replyMessage ? <ReplyPreview message={replyMessage} onCancel={onCancelReply} /> : null}
      <div className="flex items-end gap-3">
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="メッセージを入力..."
          className="h-24 flex-1 resize-none rounded-3xl border border-gray-200 bg-gray-50 px-4 py-3 text-sm leading-relaxed text-gray-900 shadow-sm focus:border-green-400 focus:bg-white focus:outline-none focus:ring-2 focus:ring-green-200"
        />
        <Button type="submit" size="lg" disabled={isSending || value.trim().length === 0}>
          {isSending ? "送信中" : "送信"}
        </Button>
      </div>
      <p className="mt-2 text-xs text-gray-400">Enterで送信 / Shift + Enterで改行</p>
    </form>
  );
};

export default ChatInput;
