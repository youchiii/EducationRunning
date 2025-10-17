import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import ChatHeader from "../components/ChatHeader";
import ChatInput from "../components/ChatInput";
import MessageList from "../components/MessageList";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";
import {
  fetchChatConversations,
  fetchChatMessages,
  markConversationRead,
  sendChatMessage,
  type ChatMessage,
  type ConversationSummary,
} from "../services/api";

const ChatRoomPage = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const params = useParams<{ conversationId: string }>();
  const conversationId = params.conversationId ? Number.parseInt(params.conversationId, 10) : Number.NaN;

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversation, setConversation] = useState<ConversationSummary | null>(null);
  const [inputValue, setInputValue] = useState("");
  const [replyTarget, setReplyTarget] = useState<ChatMessage | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadConversation = useCallback(
    async (initial = false) => {
      if (Number.isNaN(conversationId)) {
        setError("チャットIDが正しくありません。");
        setConversation(null);
        setMessages([]);
        return;
      }

      if (initial) {
        setIsLoading(true);
      } else {
        setIsRefreshing(true);
      }

      try {
        const [conversationList, messageList] = await Promise.all([
          fetchChatConversations(),
          fetchChatMessages(conversationId),
        ]);

        const found = conversationList.find((item) => item.conversation_id === conversationId) ?? null;
        if (!found) {
          setError("チャットが見つかりませんでした。");
          setConversation(null);
          setMessages([]);
          return;
        }

        setConversation({ ...found, unread_count: 0 });
        setMessages(messageList);
        setError(null);
        try {
          await markConversationRead(conversationId);
        } catch (readError) {
          console.warn("Failed to mark messages as read", readError);
        }
      } catch (requestError) {
        console.error(requestError);
        setError("チャットの読み込みに失敗しました。");
      } finally {
        if (initial) {
          setIsLoading(false);
        }
        setIsRefreshing(false);
      }
    },
    [conversationId],
  );

  useEffect(() => {
    void loadConversation(true);
  }, [loadConversation]);

  const handleSend = useCallback(async () => {
    if (!user || !conversation || !inputValue.trim()) {
      return;
    }

    setIsSending(true);
    try {
      const response = await sendChatMessage({
        sender_id: user.id,
        receiver_id: conversation.partner_id,
        text: inputValue.trim(),
        reply_to_id: replyTarget?.id ?? null,
      });
      setMessages((prev) => [...prev, response.message]);
      setConversation((prev) =>
        prev
          ? {
              ...prev,
              latest_message: {
                id: response.message.id,
                text: response.message.text,
                timestamp: response.message.timestamp,
              },
            }
          : prev,
      );
      setInputValue("");
      setReplyTarget(null);
      setError(null);
    } catch (requestError) {
      console.error(requestError);
      setError("メッセージの送信に失敗しました。");
    } finally {
      setIsSending(false);
    }
  }, [conversation, inputValue, replyTarget, user]);

  const handleReply = useCallback((message: ChatMessage) => {
    setReplyTarget(message);
  }, []);

  const handleCancelReply = useCallback(() => {
    setReplyTarget(null);
  }, []);

  const content = useMemo(() => {
    if (isLoading) {
      return (
        <div className="flex flex-1 items-center justify-center text-sm text-gray-500">読み込み中です...</div>
      );
    }
    if (!conversation) {
      return (
        <div className="flex flex-1 flex-col items-center justify-center gap-4 p-6 text-center text-sm text-gray-500">
          <p>{error ?? "チャットが表示できません。"}</p>
          <Button variant="outline" onClick={() => void loadConversation(true)}>
            再読み込み
          </Button>
        </div>
      );
    }
    return <MessageList messages={messages} currentUserId={user?.id ?? 0} onReply={handleReply} />;
  }, [conversation, error, handleReply, isLoading, loadConversation, messages, user?.id]);

  if (!user) {
    return null;
  }

  return (
    <div className="flex flex-col space-y-6">
      {error && conversation ? (
        <div className="rounded-2xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      <div className="flex min-h-[560px] flex-col overflow-hidden rounded-3xl border border-gray-200 bg-white shadow-xl">
        {conversation ? (
          <ChatHeader
            partnerName={conversation.partner_name}
            unreadCount={conversation.unread_count}
            onBack={() => navigate("/chat")}
            onRefresh={() => void loadConversation(false)}
            isRefreshing={isRefreshing}
          />
        ) : (
          <div className="border-b border-gray-200 bg-white/95 px-4 py-3 shadow-sm">
            <Button variant="ghost" size="sm" onClick={() => navigate("/chat")}>
              一覧へ戻る
            </Button>
          </div>
        )}

        {content}

        {conversation ? (
          <ChatInput
            value={inputValue}
            onChange={setInputValue}
            onSend={handleSend}
            isSending={isSending}
            replyMessage={replyTarget}
            onCancelReply={handleCancelReply}
          />
        ) : null}
      </div>
    </div>
  );
};

export default ChatRoomPage;
