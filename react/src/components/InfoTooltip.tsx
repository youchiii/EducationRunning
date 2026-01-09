import { useId, useState, type CSSProperties, type ReactNode } from "react";

type TooltipSide = "top" | "right" | "bottom" | "left";

export type InfoTooltipProps = {
  content: ReactNode;
  ariaLabel?: string;
  asInline?: boolean;
  className?: string;
  contentClassName?: string;
  side?: TooltipSide;
};

type SidePosition = {
  container: string;
  arrow: CSSProperties;
};

const SIDE_POSITIONS: Record<TooltipSide, SidePosition> = {
  top: {
    container: "bottom-full left-1/2 -translate-x-1/2 mb-2",
    arrow: { bottom: "-4px", left: "calc(50% - 4px)" },
  },
  right: {
    container: "left-full top-1/2 -translate-y-1/2 ml-2",
    arrow: { left: "-4px", top: "calc(50% - 4px)" },
  },
  bottom: {
    container: "top-full left-1/2 -translate-x-1/2 mt-2",
    arrow: { top: "-4px", left: "calc(50% - 4px)" },
  },
  left: {
    container: "right-full top-1/2 -translate-y-1/2 mr-2",
    arrow: { right: "-4px", top: "calc(50% - 4px)" },
  },
};

const INLINE_CLASS = "inline-flex items-center gap-1";
const BLOCK_CLASS = "flex items-center gap-2";

const ICON_BUTTON_BASE =
  "relative inline-flex h-6 w-6 items-center justify-center rounded-full border border-border/60 bg-muted text-muted-foreground transition hover:bg-muted/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60";

const TOOLTIP_BASE =
  "absolute z-50 rounded-xl border border-border/60 bg-background/95 text-xs leading-relaxed text-foreground shadow-lg backdrop-blur-sm text-left";

const ARIA_LABEL_FALLBACK = "詳細を表示";

const IconGlyph = () => (
  <svg
    aria-hidden="true"
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="16" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12" y2="8" />
  </svg>
);

const InfoTooltip = ({
  content,
  ariaLabel,
  asInline = false,
  className,
  contentClassName,
  side = "top",
}: InfoTooltipProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const tooltipId = useId();
  const { container, arrow } = SIDE_POSITIONS[side];

  const wrapperClassName = [asInline ? INLINE_CLASS : BLOCK_CLASS, className]
    .filter(Boolean)
    .join(" ");

  return (
    <span className={wrapperClassName}>
      <span className="relative inline-flex">
        <button
          type="button"
          className={ICON_BUTTON_BASE}
          aria-label={ariaLabel ?? ARIA_LABEL_FALLBACK}
          aria-describedby={isOpen ? tooltipId : undefined}
          onMouseEnter={() => setIsOpen(true)}
          onMouseLeave={() => setIsOpen(false)}
          onFocus={() => setIsOpen(true)}
          onBlur={() => setIsOpen(false)}
        >
          <IconGlyph />
        </button>
        {isOpen && (
          <div
            id={tooltipId}
            role="tooltip"
            className={[TOOLTIP_BASE, "px-3 py-2 whitespace-normal break-words", contentClassName, container]
              .filter(Boolean)
              .join(" ")}
            style={{
              minWidth: "220px",
              maxWidth: "280px",
              whiteSpace: "normal",
              wordBreak: "break-word",
              textAlign: "left",
              padding: "0.5rem 0.75rem",
            }}
          >
            <span
              aria-hidden="true"
              style={{
                position: "absolute",
                width: "8px",
                height: "8px",
                transform: "rotate(45deg)",
                background: "inherit",
                borderLeft: "1px solid rgba(148, 163, 184, 0.6)",
                borderTop: "1px solid rgba(148, 163, 184, 0.6)",
                ...arrow,
              }}
            />
            <span className="block text-foreground">{content}</span>
          </div>
        )}
      </span>
    </span>
  );
};

export default InfoTooltip;
