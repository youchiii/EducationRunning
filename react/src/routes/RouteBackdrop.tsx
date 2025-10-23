import { useMemo } from "react";
import { useLocation } from "react-router-dom";
import SakuraFallCanvas from "../components/SakuraFallCanvas";
import { useAuth } from "../context/AuthContext";

const TARGET_PATHS = new Set(["/", "/splash", "/login", "/signup"]);

const normalisePath = (pathname: string) => {
  if (!pathname) {
    return "/";
  }
  const trimmed = pathname.endsWith("/") && pathname !== "/" ? pathname.slice(0, -1) : pathname;
  return trimmed.toLowerCase();
};

const RouteBackdrop = () => {
  const { pathname } = useLocation();
  const { user } = useAuth();

  const shouldShow = useMemo(() => {
    if (user) {
      return false;
    }
    const cleaned = normalisePath(pathname);
    return TARGET_PATHS.has(cleaned);
  }, [pathname, user]);

  if (!shouldShow) {
    return null;
  }

  return (
    <SakuraFallCanvas
      density={80}
      maxPetals={140}
      gravity={0.08}
      wind={{ mean: 0.25, variance: 0.12 }}
      spin={{ min: -0.025, max: 0.02 }}
      zIndex={6}
      parallax
    />
  );
};

export default RouteBackdrop;
