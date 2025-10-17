import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import Header from "../components/Header";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";
import { fetchChatConversations, type ConversationSummary } from "../services/api";

const formatTimestamp = (timestamp: string | null | undefined) => {
  if (!timestamp) {
    return "メッセージはまだありません";
  }
  const value = timestamp.replace(" ", "T");
  const date = new Date(value.endsWith("Z") ? value : `${value}`);
  if (Number.isNaN(date.getTime())) {
    return "メッセージはまだありません";
  }
  const today = new Date();
  const isSameDay =
    date.getDate() === today.getDate() &&
    date.getMonth() === today.getMonth() &&
    date.getFullYear() === today.getFullYear();
  return isSameDay
    ? date.toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString("ja-JP", { month: "short", day: "numeric" });
};

const ChatListPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadConversations = useCallback(async () => {
    try {
      setIsLoading(true);
      const result = await fetchChatConversations();
      setConversations(result);
      setError(null);
    } catch (requestError) {
      console.error(requestError);
      setError("チャット一覧の取得に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadConversations();
  }, [loadConversations]);

  const emptyMessage = useMemo(() => {
    if (isLoading) {
      return "読み込み中です...";
    }
    if (error) {
      return error;
    }
    return "まだチャットはありません。右下の会話から気軽に話しかけてみましょう。";
  }, [isLoading, error]);

  if (!user) {
    return null;
  }

  return (
    <div className="space-y-6">
      <Header title="課題チャット" subtitle="LINEのような会話画面で、先生やクラスメイトと連絡を取り合えます。" />

      {error ? (
        <div className="rounded-2xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      ) : null}

      <div className="rounded-3xl border border-gray-200 bg-white p-6 shadow-xl">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm font-semibold text-gray-600">最近のチャット</p>
          <Button variant="outline" size="sm" onClick={() => void loadConversations()} disabled={isLoading}>
            更新
          </Button>
        </div>

        {conversations.length === 0 ? (
          <div className="flex h-48 flex-col items-center justify-center rounded-2xl bg-gray-50 text-center text-sm text-gray-500">
            {emptyMessage}
          </div>
        ) : (
          <ul className="space-y-3">
            {conversations.map((conversation) => {
              const latestText = conversation.latest_message?.text ?? "メッセージはまだありません";
              const latestTimestamp = conversation.latest_message?.timestamp ?? null;
              return (
                <li key={conversation.conversation_id}>
                  <button
                    type="button"
                    onClick={() => navigate(`/chat/${conversation.conversation_id}`)}
                    className="group flex w-full items-center justify-between rounded-2xl border border-transparent bg-gray-50 px-4 py-4 text-left shadow-sm transition hover:-translate-y-[1px] hover:border-green-200 hover:bg-green-50 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-green-200"
                  >
                    <div>
                      <div className="flex items-center gap-3">
                        <p className="text-base font-semibold text-gray-900">{conversation.partner_name}</p>
                        {conversation.unread_count > 0 ? (
                          <span className="inline-flex min-w-[28px] justify-center rounded-full bg-blue-500 px-2 py-0.5 text-xs font-semibold text-white">
                            {conversation.unread_count}
                          </span>
                        ) : null}
                      </div>
                      <p className="mt-1 truncate text-sm text-gray-600">
                        {latestText}
                      </p>
                    </div>
                    <span className="ml-4 text-xs font-medium text-gray-400">
                      {formatTimestamp(latestTimestamp)}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
};

export default ChatListPage;
