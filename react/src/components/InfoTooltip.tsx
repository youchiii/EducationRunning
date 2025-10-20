import {
  type CSSProperties,
  type FocusEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { Info } from "lucide-react";

const PLACEMENT_CLASSES: Record<string, string> = {
  top: "bottom-full left-1/2 -translate-x-1/2 -translate-y-2",
  right: "left-full top-1/2 -translate-y-1/2 translate-x-2",
  bottom: "top-full left-1/2 -translate-x-1/2 translate-y-2",
  left: "right-full top-1/2 -translate-y-1/2 -translate-x-2",
};

const DEFAULT_PLACEMENT = "right";

export type InfoTooltipProps = {
  label?: string;
  content: ReactNode;
  placement?: "top" | "right" | "bottom" | "left";
  iconSize?: number;
  asInline?: boolean;
  ariaLabel?: string;
};

const InfoTooltip = ({
  label,
  content,
  placement = DEFAULT_PLACEMENT,
  iconSize = 16,
  asInline = true,
  ariaLabel,
}: InfoTooltipProps) => {
  const [open, setOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const tooltipId = useId();

  const closeTooltip = useCallback(() => setOpen(false), []);
  const toggleTooltip = useCallback(() => setOpen((prev) => !prev), []);
  const [tooltipStyle, setTooltipStyle] = useState<CSSProperties | null>(null);

  const calculatePosition = useCallback(() => {
    if (!buttonRef.current || !tooltipRef.current) {
      return;
    }

    const triggerRect = buttonRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();
    const spacing = 12;
    const viewportPadding = 8;

    let top = triggerRect.top;
    let left = triggerRect.left;

    switch (placement) {
      case "top":
        top = triggerRect.top - tooltipRect.height - spacing;
        left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
        break;
      case "bottom":
        top = triggerRect.bottom + spacing;
        left = triggerRect.left + triggerRect.width / 2 - tooltipRect.width / 2;
        break;
      case "left":
        top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
        left = triggerRect.left - tooltipRect.width - spacing;
        break;
      case "right":
      default:
        top = triggerRect.top + triggerRect.height / 2 - tooltipRect.height / 2;
        left = triggerRect.right + spacing;
        break;
    }

    const maxTop = window.innerHeight - tooltipRect.height - viewportPadding;
    const maxLeft = window.innerWidth - tooltipRect.width - viewportPadding;

    const clampedTop = Math.min(Math.max(top, viewportPadding), Math.max(maxTop, viewportPadding));
    const clampedLeft = Math.min(Math.max(left, viewportPadding), Math.max(maxLeft, viewportPadding));

    setTooltipStyle({ top: clampedTop, left: clampedLeft, position: "fixed" });
  }, [placement]);

  useEffect(() => {
    if (!open) {
      setTooltipStyle(null);
      return;
    }
    calculatePosition();
    const handleResize = () => calculatePosition();
    window.addEventListener("resize", handleResize);
    window.addEventListener("scroll", handleResize, true);
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        closeTooltip();
        buttonRef.current?.focus();
      }
    };
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!tooltipRef.current || !buttonRef.current) {
        return;
      }
      if (!tooltipRef.current.contains(target) && !buttonRef.current.contains(target)) {
        closeTooltip();
      }
    };
    document.addEventListener("keydown", handleKey, true);
    document.addEventListener("mousedown", handleClickOutside, true);
    document.addEventListener("touchstart", handleClickOutside, true);
    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("scroll", handleResize, true);
      document.removeEventListener("keydown", handleKey, true);
      document.removeEventListener("mousedown", handleClickOutside, true);
      document.removeEventListener("touchstart", handleClickOutside, true);
    };
  }, [open, closeTooltip, calculatePosition]);

  const handleBlur = (event: FocusEvent<HTMLButtonElement>) => {
    const next = event.relatedTarget as Node | null;
    if (!tooltipRef.current?.contains(next)) {
      closeTooltip();
    }
  };

  const iconButton = (
    <button
      ref={buttonRef}
      type="button"
      aria-label={ariaLabel ?? (label ? `用語解説: ${label}` : "用語解説")}
      aria-describedby={open ? tooltipId : undefined}
      aria-expanded={open}
      onClick={toggleTooltip}
      onBlur={handleBlur}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggleTooltip();
        }
      }}
      className="inline-flex items-center justify-center rounded-full border border-transparent bg-muted px-1.5 py-1 text-xs text-muted-foreground transition hover:bg-muted/80 focus:outline-none focus:ring-2 focus:ring-primary/60"
      tabIndex={0}
    >
      <Info aria-hidden={true} width={iconSize} height={iconSize} />
    </button>
  );

  return (
    <div className={asInline ? "inline-flex items-center gap-1" : "flex items-center gap-2"}>
      {label && <span className="text-sm font-semibold text-foreground">{label}</span>}
      <div className="relative inline-flex items-center">
        {iconButton}
        {open && (
          <div
            ref={tooltipRef}
            id={tooltipId}
            role="tooltip"
            className={`fixed z-50 w-[min(28rem,90vw)] max-w-[90vw] max-h-[70vh] overflow-auto rounded-lg border border-border/60 bg-background p-4 text-xs text-muted-foreground shadow-xl backdrop-blur-sm`}
            style={{
              ...tooltipStyle,
              visibility: tooltipStyle ? "visible" : "hidden",
            }}
          >
            <div className="space-y-2">
              {label && <p className="text-sm font-semibold text-foreground">{label}</p>}
              <div className="leading-relaxed text-foreground/90">{content}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default InfoTooltip;
