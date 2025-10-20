import { useEffect, useRef, useState } from "react";

type PlotlyType = {
  newPlot: (element: HTMLElement, data: unknown, layout?: unknown, config?: unknown) => Promise<void> | void;
  react: (element: HTMLElement, data: unknown, layout?: unknown, config?: unknown) => Promise<void> | void;
  purge: (element: HTMLElement) => void;
};

declare global {
  interface Window {
    Plotly?: PlotlyType;
  }
}

const PLOTLY_CDN_URL = "https://cdn.plot.ly/plotly-2.30.0.min.js";
let plotlyPromise: Promise<PlotlyType> | null = null;

const loadPlotly = () => {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Plotly requires a browser environment"));
  }
  if (window.Plotly) {
    return Promise.resolve(window.Plotly);
  }
  if (!plotlyPromise) {
    plotlyPromise = new Promise<PlotlyType>((resolve, reject) => {
      const existing = document.querySelector(`script[src='${PLOTLY_CDN_URL}']`);
      if (existing) {
        existing.addEventListener("load", () => {
          if (window.Plotly) {
            resolve(window.Plotly);
          } else {
            reject(new Error("Plotly failed to load"));
          }
        });
        existing.addEventListener("error", () => reject(new Error("Plotly failed to load")));
        return;
      }
      const script = document.createElement("script");
      script.src = PLOTLY_CDN_URL;
      script.async = true;
      script.onload = () => {
        if (window.Plotly) {
          resolve(window.Plotly);
        } else {
          reject(new Error("Plotly failed to initialize"));
        }
      };
      script.onerror = () => reject(new Error("Plotly failed to load"));
      document.body.appendChild(script);
    });
  }
  return plotlyPromise;
};

type PlotlyChartProps = {
  data: Array<Record<string, unknown>>;
  layout?: Record<string, unknown>;
  config?: Record<string, unknown>;
  className?: string;
  style?: React.CSSProperties;
  loadingFallback?: React.ReactNode;
  errorFallback?: React.ReactNode;
};

const PlotlyChart = ({
  data,
  layout,
  config,
  className,
  style,
  loadingFallback,
  errorFallback,
}: PlotlyChartProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<PlotlyType | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    let isMounted = true;
    const attachPlot = async () => {
      try {
        const plotly = await loadPlotly();
        if (!isMounted) {
          return;
        }
        plotlyRef.current = plotly;
        const target = containerRef.current;
        if (!target) {
          return;
        }
        setIsLoaded(true);
        setLoadError(null);
        const mergedConfig = { responsive: true, displaylogo: false, ...config };
        if (target.dataset.plotlyInitialized) {
          await plotly.react(target, data, layout, mergedConfig);
        } else {
          await plotly.newPlot(target, data, layout, mergedConfig);
          target.dataset.plotlyInitialized = "true";
        }
      } catch (plotlyError) {
        if (!isMounted) {
          return;
        }
        console.error(plotlyError);
        setLoadError((plotlyError as Error).message);
      }
    };

    attachPlot();

    return () => {
      isMounted = false;
      const plotly = plotlyRef.current;
      const target = containerRef.current;
      if (plotly && target) {
        plotly.purge(target);
        delete target.dataset.plotlyInitialized;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, layout, config]);

  if (loadError) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-dashed border-border/60 bg-muted/20 p-6 text-sm text-destructive">
        {errorFallback ?? `Plotlyの読み込みに失敗しました: ${loadError}`}
      </div>
    );
  }

  return (
    <div className={className} style={style}>
      {!isLoaded && (
        <div className="flex items-center justify-center rounded-lg border border-dashed border-border/60 bg-muted/20 p-6 text-sm text-muted-foreground">
          {loadingFallback ?? "グラフを読み込み中..."}
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
};

export default PlotlyChart;
