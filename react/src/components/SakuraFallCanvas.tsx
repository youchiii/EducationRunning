import { type CSSProperties, useEffect, useRef } from "react";

/**
 * Configuration options for the sakura petal canvas overlay.
 */
export interface SakuraFallProps {
  density?: number;
  maxPetals?: number;
  gravity?: number;
  wind?: { mean: number; variance: number };
  spin?: { min: number; max: number };
  palette?: string[];
  zIndex?: number;
  parallax?: boolean;
}

type Petal = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number;
  spin: number;
  baseScale: number;
  depth: number;
  color: string;
  opacity: number;
  swaySpeed: number;
  swayDistance: number;
  swayOffset: number;
  shape: number;
};

const DEFAULT_PALETTE = ["#F7C4DA", "#F4AFCE", "#F0A0C6", "#D94E8F"];

const SHAPE_DRAWERS: Array<(ctx: CanvasRenderingContext2D) => void> = [
  (ctx) => {
    ctx.moveTo(0, -1);
    ctx.bezierCurveTo(0.45, -0.6, 0.55, 0.0, 0.1, 0.9);
    ctx.bezierCurveTo(-0.1, 1.0, -0.4, 0.4, -0.3, -0.4);
    ctx.bezierCurveTo(-0.2, -0.9, 0, -1, 0, -1);
  },
  (ctx) => {
    ctx.moveTo(0, -1);
    ctx.bezierCurveTo(0.6, -0.6, 0.7, 0.2, 0.1, 1);
    ctx.bezierCurveTo(-0.2, 0.8, -0.6, 0.2, -0.4, -0.5);
    ctx.bezierCurveTo(-0.3, -0.9, 0, -1, 0, -1);
  },
  (ctx) => {
    ctx.moveTo(0, -1);
    ctx.quadraticCurveTo(0.7, -0.4, 0.3, 0.6);
    ctx.quadraticCurveTo(0.1, 1.1, -0.5, 0.7);
    ctx.quadraticCurveTo(-0.7, 0.2, -0.2, -0.6);
    ctx.quadraticCurveTo(-0.1, -1, 0, -1);
  },
];

const CANVAS_STYLE: CSSProperties = {
  position: "fixed",
  inset: 0,
  pointerEvents: "none",
  width: "100%",
  height: "100%",
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const randomInRange = (min: number, max: number) => Math.random() * (max - min) + min;

const supportsPrefersReducedMotion = () => typeof window !== "undefined" && typeof window.matchMedia === "function";

const attachMediaListener = (media: MediaQueryList, handler: (event: MediaQueryListEvent | MediaQueryList) => void) => {
  if (typeof media.addEventListener === "function") {
    media.addEventListener("change", handler);
    return () => media.removeEventListener("change", handler);
  }
  // Safari < 14 fallback
  media.addListener(handler as (event: MediaQueryListEvent) => void);
  return () => media.removeListener(handler as (event: MediaQueryListEvent) => void);
};

const resolveWind = (wind?: SakuraFallProps["wind"]) => ({
  mean: wind?.mean ?? 0.2,
  variance: wind?.variance ?? 0.1,
});

const resolveSpin = (spin?: SakuraFallProps["spin"]) => {
  const min = spin?.min ?? -0.02;
  const max = spin?.max ?? 0.02;
  return min <= max ? { min, max } : { min: max, max: min };
};

const computeTargetCount = (
  width: number,
  height: number,
  density: number,
  maxPetals: number,
  reduceMotion: boolean,
) => {
  if (reduceMotion) {
    return Math.min(5, maxPetals);
  }
  const areaFactor = clamp((width * height) / (1280 * 720), 0.4, 2.8);
  const desired = Math.round(density * areaFactor);
  return clamp(desired, 0, maxPetals);
};

const SakuraFallCanvas = ({
  density = 60,
  maxPetals = 120,
  gravity = 0.08,
  wind,
  spin,
  palette,
  zIndex = 8,
  parallax = true,
}: SakuraFallProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const resolvedPalette = palette && palette.length > 0 ? palette : DEFAULT_PALETTE;
    const resolvedWind = resolveWind(wind);
    const resolvedSpin = resolveSpin(spin);

    let animationFrame: number | null = null;
    let petals: Petal[] = [];
    let width = 0;
    let height = 0;
    let lastTimestamp = performance.now();
    let frameEstimate = 16;
    let reduceMotion = false;
    let targetCount = 0;

    const media = supportsPrefersReducedMotion() ? window.matchMedia("(prefers-reduced-motion: reduce)") : undefined;
    let detachMotion: (() => void) | undefined;
    if (media) {
      reduceMotion = media.matches;
      detachMotion = attachMediaListener(media, (event) => {
        reduceMotion = event.matches;
        targetCount = computeTargetCount(width, height, density, maxPetals, reduceMotion);
        petals = petals.slice(0, targetCount);
      });
    }

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      targetCount = computeTargetCount(width, height, density, maxPetals, reduceMotion);
      petals = petals.slice(0, targetCount);
    };

    resize();

    const spawnPetal = (initialY?: number): Petal => {
      const depth = parallax ? Math.random() : 0.6;
      const sizeFactor = parallax ? 0.55 + depth * 0.8 : 1;
      const baseOpacity = 0.7 + Math.random() * 0.3;
      const swayDistance = randomInRange(6, 16) * (parallax ? 0.6 + depth : 1);
      return {
        x: Math.random() * width,
        y: initialY ?? randomInRange(-height, 0),
        vx: resolvedWind.mean * randomInRange(0.6, 1.4),
        vy: randomInRange(0.6, 1.2) * sizeFactor,
        angle: randomInRange(0, Math.PI * 2),
        spin: randomInRange(resolvedSpin.min, resolvedSpin.max),
        baseScale: sizeFactor,
        depth,
        color: resolvedPalette[Math.floor(Math.random() * resolvedPalette.length)],
        opacity: baseOpacity,
        swaySpeed: randomInRange(0.0012, 0.0024),
        swayDistance,
        swayOffset: randomInRange(0, Math.PI * 2),
        shape: Math.floor(Math.random() * SHAPE_DRAWERS.length),
      };
    };

    const recyclePetal = (petal: Petal) => {
      const depth = parallax ? Math.random() : 0.6;
      const sizeFactor = parallax ? 0.55 + depth * 0.8 : 1;
      petal.x = Math.random() * width;
      petal.y = randomInRange(-height * 0.2, 0);
      petal.vx = resolvedWind.mean * randomInRange(0.6, 1.4);
      petal.vy = randomInRange(0.6, 1.2) * sizeFactor;
      petal.angle = randomInRange(0, Math.PI * 2);
      petal.spin = randomInRange(resolvedSpin.min, resolvedSpin.max);
      petal.baseScale = sizeFactor;
      petal.depth = depth;
      petal.opacity = 0.7 + Math.random() * 0.3;
      petal.swaySpeed = randomInRange(0.0012, 0.0024);
      petal.swayDistance = randomInRange(6, 16) * (parallax ? 0.6 + depth : 1);
      petal.swayOffset = randomInRange(0, Math.PI * 2);
      petal.shape = Math.floor(Math.random() * SHAPE_DRAWERS.length);
      petal.color = resolvedPalette[Math.floor(Math.random() * resolvedPalette.length)];
    };

    const ensurePopulation = (desiredCount: number) => {
      if (petals.length > desiredCount) {
        petals = petals.slice(0, desiredCount);
      }
      while (petals.length < desiredCount) {
        petals.push(spawnPetal());
      }
    };

    const drawPetal = (ctx: CanvasRenderingContext2D, petal: Petal) => {
      ctx.save();
      ctx.globalAlpha = petal.opacity;
      ctx.translate(petal.x, petal.y);
      ctx.rotate(petal.angle);
      const size = 12 * petal.baseScale;
      ctx.scale(size, size);
      ctx.beginPath();
      SHAPE_DRAWERS[petal.shape](ctx);
      ctx.fillStyle = petal.color;
      ctx.fill();
      ctx.restore();
    };

    const animate = (timestamp: number) => {
      const delta = timestamp - lastTimestamp;
      lastTimestamp = timestamp;
      frameEstimate = frameEstimate * 0.9 + delta * 0.1;
      const deltaFactor = delta / 16.6667 || 1;

      const performancePenalty = frameEstimate > 26 ? clamp(frameEstimate / 26, 1, 3) : 1;
      const desiredCount = Math.max(
        reduceMotion ? targetCount : Math.floor(targetCount / performancePenalty),
        reduceMotion ? Math.min(3, targetCount) : Math.min(12, targetCount),
      );
      ensurePopulation(desiredCount);

      context.clearRect(0, 0, width, height);

      for (const petal of petals) {
        const depthFactor = parallax ? 0.6 + petal.depth * 0.8 : 1;
        const sway = Math.sin(timestamp * petal.swaySpeed + petal.swayOffset) * petal.swayDistance;
        petal.vy += gravity * depthFactor * deltaFactor * 0.6;
        const windInfluence = resolvedWind.mean + Math.cos(timestamp * 0.0008 + petal.swayOffset) * resolvedWind.variance;
        petal.vx += (windInfluence - petal.vx) * 0.02;
        petal.x += (petal.vx + sway * 0.02) * deltaFactor;
        petal.y += petal.vy * deltaFactor;
        petal.angle += petal.spin * deltaFactor;

        if (petal.y > height + 60 || petal.x < -60 || petal.x > width + 60) {
          recyclePetal(petal);
        } else {
          drawPetal(context, petal);
        }
      }

      if (reduceMotion && targetCount === 0) {
        context.clearRect(0, 0, width, height);
        return;
      }

      animationFrame = window.requestAnimationFrame(animate);
    };

    animationFrame = window.requestAnimationFrame(animate);

    const handleResize = () => {
      resize();
    };

    window.addEventListener("resize", handleResize);

    return () => {
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
      }
      window.removeEventListener("resize", handleResize);
      context.clearRect(0, 0, width, height);
      if (detachMotion) {
        detachMotion();
      }
    };
  }, [density, maxPetals, gravity, wind, spin, palette, parallax]);

  return <canvas ref={canvasRef} style={{ ...CANVAS_STYLE, zIndex }} aria-hidden="true" />;
};

export default SakuraFallCanvas;
