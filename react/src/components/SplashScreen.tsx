import { motion } from "framer-motion";

const logoSrc = "/sakuragaoka_logo.jpg";
const backgroundGradient = "bg-[radial-gradient(circle_at_top,_rgba(255,192,203,0.55),_transparent_60%),radial-gradient(circle_at_bottom,_rgba(255,228,225,0.65),_transparent_45%)] bg-[#fff5f7]";
const petalCount = 18;

const SplashScreen = () => {
  return (
    <div className={`relative flex min-h-screen flex-col items-center justify-center gap-10 overflow-hidden ${backgroundGradient}`}>
      {[...Array(petalCount)].map((_, index) => {
        const delay = Math.random() * 2;
        const duration = 6 + Math.random() * 4;
        const startX = Math.random() * 100;
        const rotate = Math.random() * 360;
        return (
          <motion.span
            key={`petal-${index}`}
            className="pointer-events-none absolute h-4 w-6 rounded-full bg-gradient-to-br from-rose-200 via-rose-300 to-rose-400 opacity-70"
            style={{ left: `${startX}%`, top: "-10%" }}
            initial={{ y: "-10%", rotate }}
            animate={{ y: "120%", rotate: rotate + 120 }}
            transition={{ repeat: Infinity, duration, delay, ease: "linear" }}
          />
        );
      })}

      <motion.div
        initial={{ opacity: 0, scale: 0.85 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="z-10 flex flex-col items-center gap-6"
      >
        <div className="relative flex h-44 w-44 items-center justify-center overflow-hidden rounded-full bg-white/90 shadow-[0_30px_80px_rgba(244,114,182,0.35)] ring-4 ring-rose-200/60">
          <motion.img
            src={logoSrc}
            alt="Sakuragaoka"
            className="h-32 w-32 rounded-full object-cover"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.25, duration: 0.6, ease: "easeOut" }}
          />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.6 }}
          className="text-center"
        >
          <p className="text-xs font-semibold uppercase tracking-[0.45em] text-rose-400/80">Sakuragaoka Analytics</p>
          <h1 className="mt-3 text-3xl font-semibold text-rose-500 drop-shadow-sm">アプリ構築プロトコル起動</h1>
          <p className="mt-3 text-sm text-rose-400/80">
            依存関係を解決して実行環境を組み上げています。
          </p>
        </motion.div>
      </motion.div>

      <motion.div
        className="z-10 h-2 w-56 overflow-hidden rounded-full bg-white/60 shadow-inner"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6, duration: 0.4 }}
      >
        <motion.div
          className="h-full w-full bg-gradient-to-r from-rose-400 via-rose-500 to-rose-400"
          initial={{ x: "-100%" }}
          animate={{ x: "100%" }}
          transition={{ repeat: Infinity, duration: 1.6, ease: "easeInOut" }}
        />
      </motion.div>
    </div>
  );
};

export default SplashScreen;
