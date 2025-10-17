import { motion } from "framer-motion";

const logoSrc = "/EduRun_logo.png";

const backgroundGradient = "bg-[radial-gradient(circle_at_top,_rgba(59,130,246,0.35),_transparent_55%),radial-gradient(circle_at_bottom,_rgba(15,118,110,0.25),_transparent_45%)]";

const SplashScreen = () => {
  return (
    <div className={`flex min-h-screen flex-col items-center justify-center gap-8 ${backgroundGradient} bg-background`}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="flex flex-col items-center gap-6"
      >
        <div className="relative flex h-40 w-40 items-center justify-center overflow-hidden rounded-3xl bg-white/90 shadow-2xl">
          <motion.img
            src={logoSrc}
            alt="EduRun"
            className="h-28 w-28 object-contain"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          />
        </div>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.35, duration: 0.5 }}
          className="text-center"
        >
          <p className="text-xs font-semibold uppercase tracking-[0.4em] text-muted-foreground">EduRun Studio</p>
          <h1 className="mt-2 text-3xl font-semibold text-foreground">データが読み込まれています…</h1>
          <p className="mt-3 text-sm text-muted-foreground">走力データの可視化と学習サポートの準備をしています。</p>
        </motion.div>
      </motion.div>
      <motion.div
        className="h-1.5 w-48 overflow-hidden rounded-full bg-white/20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5, duration: 0.4 }}
      >
        <motion.div
          className="h-full w-full bg-primary"
          initial={{ x: "-100%" }}
          animate={{ x: "100%" }}
          transition={{ repeat: Infinity, duration: 1.3, ease: "easeInOut" }}
        />
      </motion.div>
    </div>
  );
};

export default SplashScreen;
