import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import Header from "../components/Header";
import Sparkline from "../components/Sparkline";
import { Button } from "../components/ui/button";
import { useDataset } from "../context/DatasetContext";

const NumericMetricCard = ({
  column,
  mean,
  median,
  mode,
  variance,
  stdDev,
  getSeries,
}: {
  column: string;
  mean: number | null;
  median: number | null;
  mode: number | null;
  variance: number | null;
  stdDev: number | null;
  getSeries: (column: string) => Promise<Array<number | null>>;
}) => {
  const [series, setSeries] = useState<Array<number | null>>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let mounted = true;
    getSeries(column)
      .then((values) => {
        if (mounted) {
          setSeries(values);
          setLoaded(true);
        }
      })
      .catch(() => {
        if (mounted) {
          setLoaded(true);
        }
      });
    return () => {
      mounted = false;
    };
  }, [column, getSeries]);

  const formatNumber = (value: number | null) => {
    if (value === null || Number.isNaN(value)) {
      return "-";
    }
    return value.toFixed(2);
  };

  return (
    <motion.div
      className="rounded-xl border border-border/60 bg-background/90 p-4 shadow-sm"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-foreground">{column}</p>
        <span className="text-xs text-muted-foreground">基本統計量</span>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-muted-foreground">
        <div>
          <p>平均</p>
          <p className="text-lg font-semibold text-foreground">{formatNumber(mean)}</p>
        </div>
        <div>
          <p>中央値</p>
          <p className="text-lg font-semibold text-foreground">{formatNumber(median)}</p>
        </div>
        <div>
          <p>最頻値</p>
          <p className="text-lg font-semibold text-foreground">{mode === null ? "-" : formatNumber(mode)}</p>
        </div>
        <div>
          <p>標準偏差</p>
          <p className="text-lg font-semibold text-foreground">{formatNumber(stdDev)}</p>
        </div>
      </div>
      <div className="mt-4">
        {loaded ? <Sparkline values={series} /> : <div className="h-[60px] rounded-md bg-muted" />}
      </div>
      <p className="mt-2 text-xs text-muted-foreground">分散: {formatNumber(variance)}</p>
    </motion.div>
  );
};

const EmptyState = ({ onFileSelect }: { onFileSelect: (file: File | null) => void }) => {
  return (
    <div className="flex flex-col items-center justify-center gap-4 rounded-2xl border border-dashed border-border/60 bg-background/80 p-10 text-center">
      <h2 className="text-2xl font-semibold text-foreground">CSVデータをアップロード</h2>
      <p className="max-w-md text-sm text-muted-foreground">
        児童・生徒の走力データをアップロードすると、統計解析・回帰分析・因子分析・チャット機能などが利用できます。
      </p>
      <label className="group flex w-full max-w-sm cursor-pointer flex-col items-center gap-3 rounded-xl border border-dashed border-primary/40 bg-primary/5 px-6 py-8 text-sm text-primary transition hover:bg-primary/10">
        <span className="text-primary/80 group-hover:text-primary">ここにファイルをドロップ</span>
        <input
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(event) => onFileSelect(event.target.files?.[0] ?? null)}
        />
      </label>
      <p className="text-xs text-muted-foreground">Shift-JIS / UTF-8 のCSVに対応しています。</p>
    </div>
  );
};

const DashboardPage = () => {
  const { dataset, stats, getSeries, uploadDataset } = useDataset();
  const [localError, setLocalError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleUpload = async (file: File | null) => {
    if (!file) {
      return;
    }
    try {
      setLocalError(null);
      await uploadDataset(file);
    } catch (error) {
      console.error(error);
      setLocalError("アップロードに失敗しました。ファイル形式を確認してください。");
    }
  };

  if (!dataset || !stats) {
    return (
      <div className="space-y-4">
        <Header title="ホーム" subtitle="CSVをアップロードしてデータダッシュボードを開始しましょう。" />
        {localError && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{localError}</div>
        )}
        <EmptyState onFileSelect={handleUpload} />
      </div>
    );
  }

  const previewColumns = dataset.preview.length > 0 ? Object.keys(dataset.preview[0]) : [];

  return (
    <div className="space-y-6">
      <Header
        title="ホーム"
        subtitle="読み込んだデータのダッシュボード。各数値項目の傾向や相関を確認できます。"
      />
      {localError && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{localError}</div>
      )}

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">📋 データプレビュー</h2>
          <p className="text-sm text-muted-foreground">
            先頭 {dataset.preview.length} 行を表示中
          </p>
        </div>
        <div className="overflow-hidden rounded-xl border border-border/60 bg-background/80 shadow-sm">
          <table className="min-w-full divide-y divide-border/60 text-sm">
            <thead className="bg-muted/50">
              <tr>
                {previewColumns.map((column) => (
                  <th key={column} className="px-4 py-3 text-left font-semibold text-muted-foreground">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataset.preview.map((row, index) => (
                <tr key={index} className="border-t border-border/40">
                  {previewColumns.map((column) => (
                    <td key={column} className="px-4 py-2 text-foreground">
                      {String(row[column] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">📈 数値項目の基本統計</h2>
          <p className="text-sm text-muted-foreground">平均・中央値・最頻値・標準偏差を確認</p>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {stats.numeric_columns.map((column) => {
            const basic = stats.basic_statistics[column];
            if (!basic) {
              return null;
            }
            return (
              <NumericMetricCard
                key={column}
                column={column}
                mean={basic.mean}
                median={basic.median}
                mode={basic.mode}
                variance={basic.variance}
                stdDev={basic.std_dev}
                getSeries={getSeries}
              />
            );
          })}
        </div>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-foreground">📊 相関係数マトリクス</h2>
        {stats.numeric_columns.length === 0 ? (
          <p className="text-sm text-muted-foreground">数値列が存在しません。</p>
        ) : (
          <div className="overflow-auto rounded-xl border border-border/60">
            <table className="min-w-full divide-y divide-border/40 text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="px-3 py-2 text-left text-muted-foreground">項目</th>
                  {stats.numeric_columns.map((column) => (
                    <th key={column} className="px-3 py-2 text-center text-muted-foreground">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {stats.numeric_columns.map((rowKey) => (
                  <tr key={rowKey} className="border-t border-border/40">
                    <td className="px-3 py-2 font-medium text-foreground">{rowKey}</td>
                    {stats.numeric_columns.map((colKey) => {
                      const value = stats.correlation_matrix[rowKey]?.[colKey];
                      const formatted = typeof value === "number" ? value.toFixed(2) : "-";
                      return (
                        <td key={colKey} className="px-3 py-2 text-center text-muted-foreground">
                          {formatted}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="rounded-xl border border-border/60 bg-background/80 p-6">
        <h2 className="text-lg font-semibold text-foreground">💡 次のステップ</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          サイドバーから統計解析・重回帰分析・因子分析・骨格推定を選択して、より詳細な分析を行いましょう。
        </p>
        <div className="mt-4 flex flex-wrap gap-3">
          <Button variant="secondary" size="sm" onClick={() => navigate("/statistics")}>統計解析へ進む</Button>
          <Button variant="ghost" size="sm" onClick={() => navigate("/regression")}>重回帰分析へ進む</Button>
          <Button variant="ghost" size="sm" onClick={() => navigate("/factor")}>因子分析へ進む</Button>
        </div>
      </section>
    </div>
  );
};

export default DashboardPage;
