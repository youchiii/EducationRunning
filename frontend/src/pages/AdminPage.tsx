import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import Header from "../components/Header";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";
import {
  deleteUser,
  fetchAllUsers,
  fetchPendingUsers,
  resetUserPassword,
  updateUserStatus,
  type UserSummary,
} from "../services/api";

const AdminPage = () => {
  const { user } = useAuth();
  const [pendingUsers, setPendingUsers] = useState<UserSummary[]>([]);
  const [allUsers, setAllUsers] = useState<UserSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [initialised, setInitialised] = useState(false);
  const [passwordUpdates, setPasswordUpdates] = useState<Record<number, string>>({});

  useEffect(() => {
    if (user?.role !== "teacher" || initialised) {
      return;
    }
    let active = true;

    const load = async () => {
      try {
        setIsLoading(true);
        const [pending, all] = await Promise.all([fetchPendingUsers(), fetchAllUsers()]);
        if (active) {
          setPendingUsers(pending);
          setAllUsers(all);
          setInitialised(true);
        }
      } catch (requestError) {
        console.error(requestError);
        if (active) {
          setError("ユーザー情報の取得に失敗しました。");
        }
      } finally {
        if (active) {
          setIsLoading(false);
        }
      }
    };

    void load();

    return () => {
      active = false;
    };
  }, [user?.role, initialised]);

  if (!user || user.role !== "teacher") {
    return (
      <div className="space-y-4">
        <Header title="ユーザー管理" subtitle="管理者専用ページです。" />
        <p className="rounded-lg border border-dashed border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
          教員アカウントでログインすると、ユーザーの承認・パスワードリセットが利用できます。
        </p>
      </div>
    );
  }

  const handleStatusUpdate = async (userId: number, status: "active" | "rejected") => {
    try {
      setIsLoading(true);
      await updateUserStatus(userId, status);
      const [pending, all] = await Promise.all([fetchPendingUsers(), fetchAllUsers()]);
      setPendingUsers(pending);
      setAllUsers(all);
    } catch (requestError) {
      console.error(requestError);
      setError("ステータスの更新に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteUser = async (userId: number) => {
    try {
      setIsLoading(true);
      await deleteUser(userId);
      const [pending, all] = await Promise.all([fetchPendingUsers(), fetchAllUsers()]);
      setPendingUsers(pending);
      setAllUsers(all);
    } catch (requestError) {
      console.error(requestError);
      setError("ユーザーの削除に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetPassword = async (event: FormEvent<HTMLFormElement>, userId: number) => {
    event.preventDefault();
    const newPassword = passwordUpdates[userId];
    if (!newPassword) {
      setError("新しいパスワードを入力してください。");
      return;
    }
    try {
      setIsLoading(true);
      await resetUserPassword(userId, newPassword);
      setPasswordUpdates((prev) => ({ ...prev, [userId]: "" }));
    } catch (requestError) {
      console.error(requestError);
      setError("パスワードのリセットに失敗しました。");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Header
        title="ユーザー管理"
        subtitle="承認待ちのユーザーを管理し、必要に応じてパスワードをリセットします。"
      />
      {error && <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{error}</div>}

      <section className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">承認待ちユーザー</h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={async () => {
              setIsLoading(true);
              try {
                const [pending, all] = await Promise.all([fetchPendingUsers(), fetchAllUsers()]);
                setPendingUsers(pending);
                setAllUsers(all);
              } finally {
                setIsLoading(false);
              }
            }}
          >
            更新
          </Button>
        </div>
        {isLoading ? (
          <p className="text-sm text-muted-foreground">読み込み中...</p>
        ) : pendingUsers.length === 0 ? (
          <p className="text-sm text-muted-foreground">承認待ちのユーザーはいません。</p>
        ) : (
          <div className="space-y-3">
            {pendingUsers.map((pending) => (
              <div key={pending.id} className="flex items-center justify-between rounded-xl border border-border/60 bg-background px-4 py-3">
                <div>
                  <p className="text-sm font-semibold text-foreground">{pending.username}</p>
                  <p className="text-xs text-muted-foreground">登録日: {pending.created_at}</p>
                </div>
                <div className="flex gap-2">
                  <Button size="sm" onClick={() => handleStatusUpdate(pending.id, "active")}>承認</Button>
                  <Button size="sm" variant="secondary" onClick={() => handleStatusUpdate(pending.id, "rejected")}>
                    却下
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-foreground">ユーザー一覧</h2>
        <div className="overflow-auto rounded-xl border border-border/60">
          <table className="min-w-full divide-y divide-border/60 text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="px-3 py-2 text-left text-muted-foreground">ユーザー名</th>
                <th className="px-3 py-2 text-left text-muted-foreground">役割</th>
                <th className="px-3 py-2 text-left text-muted-foreground">ステータス</th>
                <th className="px-3 py-2 text-left text-muted-foreground">登録日</th>
                <th className="px-3 py-2 text-left text-muted-foreground">パスワードリセット</th>
                <th className="px-3 py-2 text-left text-muted-foreground">アクション</th>
              </tr>
            </thead>
            <tbody>
              {allUsers.map((summary) => (
                <tr key={summary.id} className="border-t border-border/40">
                  <td className="px-3 py-2 text-foreground">{summary.username}</td>
                  <td className="px-3 py-2 capitalize text-muted-foreground">{summary.role}</td>
                  <td className="px-3 py-2 text-muted-foreground">{summary.status}</td>
                  <td className="px-3 py-2 text-muted-foreground">{summary.created_at}</td>
                  <td className="px-3 py-2">
                    <form className="flex items-center gap-2" onSubmit={(event) => handleResetPassword(event, summary.id)}>
                      <input
                        type="password"
                        value={passwordUpdates[summary.id] ?? ""}
                        onChange={(event) => setPasswordUpdates((prev) => ({ ...prev, [summary.id]: event.target.value }))}
                        placeholder="新しいパスワード"
                        className="w-44 rounded-lg border border-border/60 bg-background px-2 py-1 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                      <Button type="submit" size="sm" variant="secondary">
                        変更
                      </Button>
                    </form>
                  </td>
                  <td className="px-3 py-2">
                    {summary.status === "rejected" ? (
                      <div className="flex flex-wrap gap-2">
                        <Button size="sm" onClick={() => handleStatusUpdate(summary.id, "active")}>
                          承認
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="border-destructive/70 text-destructive hover:bg-destructive/10"
                          onClick={() => handleDeleteUser(summary.id)}
                        >
                          削除
                        </Button>
                      </div>
                    ) : (
                      <span className="text-xs text-muted-foreground">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};

export default AdminPage;
