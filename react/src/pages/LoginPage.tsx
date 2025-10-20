import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import { useAuth } from "../context/AuthContext";

const logoSrc = "/sakuragaoka_logo.jpg";
const backgroundGradient = "bg-[radial-gradient(circle_at_top,_rgba(255,192,203,0.55),_transparent_60%),radial-gradient(circle_at_bottom,_rgba(255,228,225,0.65),_transparent_45%)] bg-[#fff5f7]";

const LoginPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const petals = useMemo(() => Array.from({ length: 18 }), []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);
    try {
      await login({ username, password });
      navigate("/dashboard", { replace: true });
    } catch (requestError) {
      setError("ログインに失敗しました。ユーザー名とパスワードを確認してください。");
    } finally {
      setIsSubmitting(false);
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
            key={`login-petal-${index}`}
            className="pointer-events-none absolute h-4 w-6 rounded-full bg-gradient-to-br from-rose-200 via-rose-300 to-rose-400 opacity-70"
            style={{ left: `${startX}%`, top: "-10%" }}
            initial={{ y: "-10%", rotate }}
            animate={{ y: "120%", rotate: rotate + 120 }}
            transition={{ repeat: Infinity, duration, delay, ease: "linear" }}
          />
        );
      })}
      <motion.div
        className="z-10 w-full max-w-md space-y-6 rounded-3xl border border-rose-200/60 bg-white/85 p-8 shadow-[0_20px_60px_rgba(244,114,182,0.35)] backdrop-blur"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: "easeOut" }}
      >
        <div className="space-y-2 text-center">
          <div className="mx-auto h-20 w-20 overflow-hidden rounded-full border-4 border-rose-200/70 shadow-md">
            <img src={logoSrc} alt="Sakuragaoka" className="h-full w-full object-cover" />
          </div>
          <h1 className="text-3xl font-semibold text-foreground">Sakuragaoka Analytics</h1>
          <p className="text-sm text-muted-foreground">桜色のダッシュボードへログインしましょう。</p>
        </div>
        {error && <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">ユーザー名</label>
            <input
              type="text"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              placeholder="teacher"
              autoComplete="username"
              required
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">パスワード</label>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              placeholder="••••••••"
              autoComplete="current-password"
              required
            />
          </div>
          <Button type="submit" className="w-full" disabled={isSubmitting}>
            {isSubmitting ? "ログイン中..." : "ログイン"}
          </Button>
        </form>
        <p className="text-center text-sm text-muted-foreground">
          初めての方は
          <Link to="/signup" className="ml-1 text-primary underline-offset-4 hover:underline">
            サインアップ
          </Link>
        </p>
      </motion.div>
    </div>
  );
};

export default LoginPage;
