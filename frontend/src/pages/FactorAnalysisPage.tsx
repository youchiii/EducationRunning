import { useState } from "react";
import type { FormEvent } from "react";
import Header from "../components/Header";
import { Button } from "../components/ui/button";
import { useDataset } from "../context/DatasetContext";
import { runFactorAnalysis, type FactorAnalysisResponse } from "../services/api";

const FactorAnalysisPage = () => {
  const { dataset, stats } = useDataset();
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [components, setComponents] = useState(2);
  const [result, setResult] = useState<FactorAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  if (!dataset || !stats) {
    return (
      <div className="space-y-4">
        <Header title="因子分析" subtitle="まずはCSVデータをアップロードしてください。" />
        <p className="rounded-lg border border-dashed border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
          数値列を複数選択すると因子分析を実行できます。
        </p>
      </div>
    );
  }

  const handleToggle = (column: string) => {
    setSelectedColumns((prev) => (prev.includes(column) ? prev.filter((item) => item !== column) : [...prev, column]));
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (selectedColumns.length < 2) {
      setError("2つ以上の列を選択してください。");
      return;
    }
    if (components >= selectedColumns.length) {
      setError("因子数は選択した列数より少なく設定してください。");
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const response = await runFactorAnalysis(dataset.dataset_id, selectedColumns, components);
      setResult(response);
    } catch (requestError) {
      console.error(requestError);
      setError("因子分析の実行に失敗しました。データを確認してください。");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Header
        title="因子分析"
        subtitle="複数の観測変数から潜在因子を抽出し、データの構造を把握します。"
      />
      {error && <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{error}</div>}

      <form className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm" onSubmit={handleSubmit}>
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">分析する数値列</label>
          <div className="grid gap-2 md:grid-cols-2">
            {stats.numeric_columns.map((column) => (
              <label key={column} className="flex items-center gap-3 rounded-lg border border-border/60 bg-background px-3 py-2 text-sm">
                <input
                  type="checkbox"
                  checked={selectedColumns.includes(column)}
                  onChange={() => handleToggle(column)}
                  className="h-4 w-4 rounded border-border accent-primary"
                />
                <span>{column}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">
            因子の数 ({components})
          </label>
          <input
            type="range"
            min={1}
            max={Math.max(1, selectedColumns.length - 1)}
            value={components}
            onChange={(event) => setComponents(Number(event.target.value))}
            className="w-full"
          />
        </div>
        <Button type="submit" disabled={isLoading}>
          {isLoading ? "計算中..." : "因子分析を実行"}
        </Button>
      </form>

      {result && (
        <div className="space-y-4 rounded-2xl border border-primary/30 bg-primary/5 p-6 shadow-sm">
          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">因子負荷量</p>
            <div className="overflow-auto rounded-lg border border-border/60">
              <table className="min-w-full divide-y divide-border/60 text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="px-3 py-2 text-left text-muted-foreground">変数</th>
                    {Array.from({ length: result.n_components }, (_, idx) => (
                      <th key={idx} className="px-3 py-2 text-right text-muted-foreground">
                        因子{idx + 1}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.factor_loadings.map((row) => (
                    <tr key={row.variable as string} className="border-t border-border/40">
                      <td className="px-3 py-2 text-foreground">{row.variable}</td>
                      {Array.from({ length: result.n_components }, (_, idx) => (
                        <td key={idx} className="px-3 py-2 text-right text-muted-foreground">
                          {Number(row[`factor_${idx + 1}`]).toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">因子得点プレビュー</p>
            <div className="overflow-auto rounded-lg border border-border/60">
              <table className="min-w-full divide-y divide-border/60 text-sm">
                <thead className="bg-muted/50">
                  <tr>
                    <th className="px-3 py-2 text-left text-muted-foreground">Index</th>
                    {Array.from({ length: result.n_components }, (_, idx) => (
                      <th key={idx} className="px-3 py-2 text-right text-muted-foreground">
                        因子{idx + 1}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.factor_scores_preview.map((row) => (
                    <tr key={row.index} className="border-t border-border/40">
                      <td className="px-3 py-2 text-foreground">{row.index}</td>
                      {Array.from({ length: result.n_components }, (_, idx) => (
                        <td key={idx} className="px-3 py-2 text-right text-muted-foreground">
                          {Number(row[`factor_${idx + 1}`]).toFixed(3)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-sm font-medium text-foreground">寄与率</p>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg bg-background/60 p-4">
                <p className="text-xs text-muted-foreground">各因子の寄与率</p>
                <p className="mt-1 text-sm text-foreground">
                  {result.explained_variance_ratio.map((value, idx) => `因子${idx + 1}: ${(value * 100).toFixed(1)}%`).join(" / ")}
                </p>
              </div>
              <div className="rounded-lg bg-background/60 p-4">
                <p className="text-xs text-muted-foreground">累積寄与率</p>
                <p className="mt-1 text-sm text-foreground">
                  {result.cumulative_variance_ratio.map((value, idx) => `因子${idx + 1}: ${(value * 100).toFixed(1)}%`).join(" / ")}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FactorAnalysisPage;
