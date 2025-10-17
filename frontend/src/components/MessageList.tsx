import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef } from "react";

import ChatBubble from "./ChatBubble";
import type { ChatMessage } from "../services/api";

interface MessageListProps {
  messages: ChatMessage[];
  currentUserId: number;
  onReply?: (message: ChatMessage) => void;
}

const MessageList = ({ messages, currentUserId, onReply }: MessageListProps) => {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  const messageLookup = useMemo(() => {
    const map = new Map<number, ChatMessage>();
    for (const item of messages) {
      map.set(item.id, item);
    }
    return map;
  }, [messages]);

  return (
    <div className="flex-1 space-y-4 overflow-y-auto bg-white px-4 py-6 md:px-8">
      <AnimatePresence initial={false}>
        {messages.map((message) => {
          const isOwn = message.sender_id === currentUserId;
          const replyMessage = message.reply_to_id ? messageLookup.get(message.reply_to_id) ?? null : null;
          return (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
              className={isOwn ? "flex justify-end" : "flex justify-start"}
            >
              <ChatBubble message={message} isOwn={isOwn} replyMessage={replyMessage ?? undefined} onReply={onReply} />
            </motion.div>
          );
        })}
      </AnimatePresence>
      <div ref={bottomRef} />
    </div>
  );
};

export default MessageList;
