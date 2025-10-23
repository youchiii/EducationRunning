import { niceNumber, nicePercentage } from "../../utils/regression";

type SummaryCardsProps = {
  r2: number | null;
  adjustedR2: number | null;
  mae: number | null;
  mape: number | null;
  durbinWatson: number | null;
  n: number;
};

const METRIC_STYLES = "rounded-2xl border border-border/60 bg-background/90 p-4 shadow-sm";

export const SummaryCards = ({ r2, adjustedR2, mae, mape, durbinWatson, n }: SummaryCardsProps) => {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <div className={METRIC_STYLES}>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">決定係数 (R²)</p>
        <p className="mt-2 text-2xl font-semibold text-foreground">{niceNumber(r2, 3)}</p>
      </div>
      <div className={METRIC_STYLES}>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">調整済み R²</p>
        <p className="mt-2 text-2xl font-semibold text-foreground">{niceNumber(adjustedR2, 3)}</p>
      </div>
      <div className={METRIC_STYLES}>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">平均絶対誤差 (MAE)</p>
        <p className="mt-2 text-2xl font-semibold text-foreground">{niceNumber(mae, 2)}</p>
      </div>
      <div className={METRIC_STYLES}>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">平均絶対パーセント誤差 (MAPE)</p>
        <p className="mt-2 text-2xl font-semibold text-foreground">{mape === null ? "—" : nicePercentage(mape, 1)}</p>
      </div>
      <div className={METRIC_STYLES}>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">Durbin–Watson</p>
        <p className="mt-2 text-2xl font-semibold text-foreground">{niceNumber(durbinWatson, 2)}</p>
      </div>
      <div className={METRIC_STYLES}>
        <p className="text-xs uppercase tracking-wider text-muted-foreground">サンプル数</p>
        <p className="mt-2 text-2xl font-semibold text-foreground">{n.toLocaleString()}</p>
      </div>
    </div>
  );
};
