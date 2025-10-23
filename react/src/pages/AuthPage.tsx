import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useNavigate, useSearchParams } from "react-router-dom";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";

type AuthMode = "login" | "signup";

const logoSrc = "/sakuragaoka_logo.jpg";
const backgroundGradient =
  "bg-[radial-gradient(circle_at_top,_rgba(255,192,203,0.55),_transparent_60%),radial-gradient(circle_at_bottom,_rgba(255,228,225,0.65),_transparent_45%)] bg-[#fff5f7]";

const AuthPage = ({ initialMode = "login" }: { initialMode?: AuthMode }) => {
  const navigate = useNavigate();
  const { login, signup } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();

  const queryMode = searchParams.get("mode");
  const modeFromQuery: AuthMode | null =
    queryMode === "signup" ? "signup" : queryMode === "login" ? "login" : null;

  const resolveInitialMode = () => modeFromQuery ?? initialMode;

  const [mode, setMode] = useState<AuthMode>(resolveInitialMode);

  const petals = useMemo(() => Array.from({ length: 24 }), []);

  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loginSubmitting, setLoginSubmitting] = useState(false);

  const [signupUsername, setSignupUsername] = useState("");
  const [signupPassword, setSignupPassword] = useState("");
  const [signupConfirmPassword, setSignupConfirmPassword] = useState("");
  const [signupError, setSignupError] = useState<string | null>(null);
  const [signupMessage, setSignupMessage] = useState<string | null>(null);
  const [signupSubmitting, setSignupSubmitting] = useState(false);

  useEffect(() => {
    if (modeFromQuery && modeFromQuery !== mode) {
      setMode(modeFromQuery);
    } else if (!modeFromQuery && initialMode !== mode) {
      setMode(initialMode);
    }
  }, [modeFromQuery, initialMode, mode]);

  const switchMode = (nextMode: AuthMode) => {
    if (nextMode === mode) {
      return;
    }
    setMode(nextMode);
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      if (nextMode === "signup") {
        params.set("mode", "signup");
      } else {
        params.delete("mode");
      }
      return params;
    });
    if (nextMode === "login") {
      setSignupError(null);
      setSignupMessage(null);
      setSignupSubmitting(false);
    } else {
      setLoginError(null);
      setLoginSubmitting(false);
    }
  };

  const handleLoginSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setLoginError(null);
    setLoginSubmitting(true);
    try {
      await login({ username: loginUsername, password: loginPassword });
      navigate("/dashboard", { replace: true });
    } catch {
      setLoginError("ログインに失敗しました。ユーザー名とパスワードを確認してください。");
    } finally {
      setLoginSubmitting(false);
    }
  };

  const handleSignupSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSignupError(null);
    setSignupMessage(null);

    if (signupPassword !== signupConfirmPassword) {
      setSignupError("パスワードが一致しません。");
      return;
    }

    setSignupSubmitting(true);
    try {
      const message = await signup({ username: signupUsername, password: signupPassword });
      setSignupMessage(message);
      setSignupUsername("");
      setSignupPassword("");
      setSignupConfirmPassword("");
    } catch {
      setSignupError("登録に失敗しました。別のユーザー名をお試しください。");
    } finally {
      setSignupSubmitting(false);
    }
  };

  return (
    <div className={`relative flex min-h-screen items-center justify-center overflow-hidden px-4 ${backgroundGradient}`}>
      {petals.map((_, index) => {
        const delay = Math.random() * 2;
        const duration = 6 + Math.random() * 4;
        const startX = Math.random() * 100;
        const rotate = Math.random() * 360;
        return (
          <motion.span
            key={`auth-petal-${index}`}
            className="pointer-events-none absolute h-4 w-6 rounded-full bg-gradient-to-br from-rose-200 via-rose-300 to-rose-400 opacity-70"
            style={{ left: `${startX}%`, top: "-10%" }}
            initial={{ y: "-10%", rotate }}
            animate={{ y: "120%", rotate: rotate + 120 }}
            transition={{ repeat: Infinity, duration, delay, ease: "linear" }}
          />
        );
      })}

      <motion.div
        layout
        className="z-10 w-full max-w-md space-y-6 rounded-3xl border border-rose-200/60 bg-white/85 p-8 shadow-[0_20px_60px_rgba(244,114,182,0.35)] backdrop-blur"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      >
        <div className="space-y-4 text-center">
          <div className="mx-auto h-20 w-20 overflow-hidden rounded-full border-4 border-rose-200/70 shadow-md">
            <img src={logoSrc} alt="Sakuragaoka" className="h-full w-full object-cover" />
          </div>
          <h1 className="text-4xl font-semibold uppercase tracking-[0.32em] text-rose-500 drop-shadow">SAKURAGAOKA ANALYTICS</h1>
          <div className="min-h-[1.25rem]">
            <AnimatePresence mode="wait" initial={false}>
              {mode === "signup" ? (
                <motion.p
                  key="signup-tagline"
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="text-sm font-semibold uppercase tracking-[0.4em] text-rose-400/80"
                >
                  新規アカウント登録
                </motion.p>
              ) : (
                <motion.p
                  key="login-tagline"
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                  className="text-sm font-semibold uppercase tracking-[0.4em] text-rose-400/40"
                >
                  &nbsp;
                </motion.p>
              )}
            </AnimatePresence>
          </div>
          <p className="text-sm text-rose-400/80">
            {mode === "login" ? "ログインして分析をしてみよう！" : "クラブメンバーとして分析の世界に飛び込もう。"}
          </p>
        </div>

        <AnimatePresence mode="wait" initial={false}>
          {mode === "login" ? (
            <motion.div
              key="login"
              layout
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.24, ease: "easeOut" }}
              className="space-y-4"
            >
              {loginError && (
                <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{loginError}</div>
              )}
              <form className="space-y-4" onSubmit={handleLoginSubmit}>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">ユーザー名</label>
                  <input
                    type="text"
                    value={loginUsername}
                    onChange={(event) => setLoginUsername(event.target.value)}
                    className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 placeholder:text-rose-300 placeholder:opacity-80"
                    placeholder="2025_2413_yamada_tarou"
                    autoComplete="username"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">パスワード</label>
                  <input
                    type="password"
                    value={loginPassword}
                    onChange={(event) => setLoginPassword(event.target.value)}
                    className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 placeholder:text-rose-300 placeholder:opacity-80"
                    placeholder="••••••••"
                    autoComplete="current-password"
                    required
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full bg-rose-500 text-white shadow-[0_12px_32px_rgba(244,114,182,0.35)] transition-transform hover:-translate-y-0.5 hover:bg-rose-500/90 focus-visible:ring-rose-400 focus-visible:ring-offset-rose-50 disabled:bg-rose-400/80"
                  disabled={loginSubmitting}
                >
                  {loginSubmitting ? "ログイン中..." : "ログイン"}
                </Button>
              </form>
            </motion.div>
          ) : (
            <motion.div
              key="signup"
              layout
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.24, ease: "easeOut" }}
              className="space-y-4"
            >
              {signupError && (
                <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{signupError}</div>
              )}
              {signupMessage && (
                <div className="rounded-lg border border-rose-300/60 bg-rose-100/40 p-3 text-sm text-rose-500">{signupMessage}</div>
              )}
              <form className="space-y-4" onSubmit={handleSignupSubmit}>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">ユーザー名</label>
                  <input
                    type="text"
                    value={signupUsername}
                    onChange={(event) => setSignupUsername(event.target.value)}
                    className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 placeholder:text-rose-300 placeholder:opacity-80"
                    placeholder="new_member_2025"
                    required
                    minLength={3}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">パスワード</label>
                  <input
                    type="password"
                    value={signupPassword}
                    onChange={(event) => setSignupPassword(event.target.value)}
                    className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 placeholder:text-rose-300 placeholder:opacity-80"
                    placeholder="••••••••"
                    required
                    minLength={6}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">パスワード（確認）</label>
                  <input
                    type="password"
                    value={signupConfirmPassword}
                    onChange={(event) => setSignupConfirmPassword(event.target.value)}
                    className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40 placeholder:text-rose-300 placeholder:opacity-80"
                    placeholder="再入力してください"
                    required
                    minLength={6}
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full bg-rose-500 text-white shadow-[0_12px_32px_rgba(244,114,182,0.35)] transition-transform hover:-translate-y-0.5 hover:bg-rose-500/90 focus-visible:ring-rose-400 focus-visible:ring-offset-rose-50 disabled:bg-rose-400/80"
                  disabled={signupSubmitting}
                >
                  {signupSubmitting ? "登録中..." : "登録する"}
                </Button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>

        <p className="text-center text-sm text-rose-400/80">
          {mode === "login" ? (
            <>
              初めての方は
              <button
                type="button"
                className="ml-1 text-rose-500 underline-offset-4 hover:underline"
                onClick={() => switchMode("signup")}
              >
                サインアップ
              </button>
            </>
          ) : (
            <>
              すでにアカウントをお持ちですか？
              <button
                type="button"
                className="ml-1 text-rose-500 underline-offset-4 hover:underline"
                onClick={() => switchMode("login")}
              >
                ログイン
              </button>
            </>
          )}
        </p>
      </motion.div>
    </div>
  );
};

export default AuthPage;
