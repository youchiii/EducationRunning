import { useEffect, useState } from "react";

const getInitialMode = () => {
  if (typeof document === "undefined") {
    return false;
  }
  if (document.documentElement.classList.contains("dark")) {
    return true;
  }
  if (typeof window !== "undefined" && window.matchMedia) {
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  return false;
};

export const useIsDarkMode = () => {
  const [isDark, setIsDark] = useState<boolean>(getInitialMode);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const updateMode = () => {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const hasDarkClass = document.documentElement.classList.contains("dark");
      setIsDark(prefersDark || hasDarkClass);
    };

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", updateMode);

    const observer = new MutationObserver(updateMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });

    updateMode();

    return () => {
      media.removeEventListener("change", updateMode);
      observer.disconnect();
    };
  }, []);

  return isDark;
};

