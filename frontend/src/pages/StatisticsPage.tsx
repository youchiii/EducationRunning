import { useMemo, useState } from "react";
import Header from "../components/Header";
import { Button } from "../components/ui/button";
import { useDataset } from "../context/DatasetContext";
import { runChiSquare, runTTest, type ChiSquareResponse, type TTestResponse } from "../services/api";

const StatisticsPage = () => {
  const { dataset, stats, refreshStats } = useDataset();
  const [chiColumnA, setChiColumnA] = useState("");
  const [chiColumnB, setChiColumnB] = useState("");
  const [chiResult, setChiResult] = useState<ChiSquareResponse | null>(null);
  const [chiLoading, setChiLoading] = useState(false);
  const [tNumeric, setTNumeric] = useState("");
  const [tGroup, setTGroup] = useState("");
  const [groupA, setGroupA] = useState("");
  const [groupB, setGroupB] = useState("");
  const [tResult, setTResult] = useState<TTestResponse | null>(null);
  const [tLoading, setTLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const categoricalOptions = dataset?.categorical_columns ?? [];
  const numericOptions = stats?.numeric_columns ?? [];

  const previewGroups = useMemo(() => {
    if (!dataset || !tGroup) {
      return [];
    }
    const unique = new Set<string>();
    dataset.preview.forEach((row) => {
      const value = row[tGroup];
      if (typeof value === "string" || typeof value === "number") {
        unique.add(String(value));
      }
    });
    return Array.from(unique).slice(0, 6);
  }, [dataset, tGroup]);

  const handleChiSquare = async () => {
    if (!dataset || !chiColumnA || !chiColumnB) {
      setError("χ²検定に必要な列が選択されていません。");
      return;
    }
    setError(null);
    setChiLoading(true);
    try {
      const response = await runChiSquare(dataset.dataset_id, chiColumnA, chiColumnB);
      setChiResult(response);
    } catch (requestError) {
      console.error(requestError);
      setError("χ²検定の実行に失敗しました。");
    } finally {
      setChiLoading(false);
    }
  };

  const handleTTest = async () => {
    if (!dataset || !tNumeric || !tGroup) {
      setError("t検定に必要な列が選択されていません。");
      return;
    }
    setError(null);
    setTLoading(true);
    try {
      const groups = groupA && groupB ? [groupA, groupB] : undefined;
      const response = await runTTest(dataset.dataset_id, tNumeric, tGroup, groups);
      setTResult(response);
    } catch (requestError) {
      console.error(requestError);
      setError("t検定の実行に失敗しました。");
    } finally {
      setTLoading(false);
    }
  };

  if (!dataset || !stats) {
    return (
      <div className="space-y-4">
        <Header title="統計解析" subtitle="まずはCSVデータをアップロードしてください。" />
        <p className="rounded-lg border border-dashed border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
          データが読み込まれるとχ²検定・t検定を実行できます。
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <Header
        title="統計解析"
        subtitle="カテゴリ間の関係をχ²検定で検証し、2群間の平均差をt検定で確認します。"
      />
      {error && <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{error}</div>}

      <section className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">χ²検定</h2>
          <Button variant="ghost" size="sm" onClick={refreshStats}>統計量を更新</Button>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">カテゴリ列①</label>
            <select
              value={chiColumnA}
              onChange={(event) => setChiColumnA(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option value="">選択してください</option>
              {categoricalOptions.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">カテゴリ列②</label>
            <select
              value={chiColumnB}
              onChange={(event) => setChiColumnB(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option value="">選択してください</option>
              {categoricalOptions.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
        </div>
        <Button onClick={handleChiSquare} disabled={chiLoading}>
          {chiLoading ? "計算中..." : "χ²検定を実行"}
        </Button>
        {chiResult && (
          <div className="space-y-4 rounded-xl border border-primary/30 bg-primary/5 p-4">
            <p className="text-sm text-muted-foreground">
              χ²統計量: <span className="font-semibold text-foreground">{chiResult.chi2.toFixed(3)}</span> / p値: {chiResult.p_value.toFixed(4)}
            </p>
            <p className="text-sm text-muted-foreground">
              判定: {chiResult.significant ? <span className="font-semibold text-primary">有意な差あり (p &lt; 0.05)</span> : "有意な差なし"}
            </p>
            <div className="overflow-auto rounded-lg border border-border/60">
              <table className="min-w-full divide-y divide-border/60 text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    {Object.keys(chiResult.contingency_table[0] ?? {}).map((column) => (
                      <th key={column} className="px-3 py-2 text-left font-semibold text-muted-foreground">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {chiResult.contingency_table.map((row, index) => (
                    <tr key={index} className="border-t border-border/40">
                      {Object.entries(row).map(([key, value]) => (
                        <td key={key} className="px-3 py-2 text-foreground">
                          {typeof value === "number" ? value.toString() : String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-foreground">t検定（2群）</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">数値列</label>
            <select
              value={tNumeric}
              onChange={(event) => setTNumeric(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option value="">選択してください</option>
              {numericOptions.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">グループ列</label>
            <select
              value={tGroup}
              onChange={(event) => setTGroup(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option value="">選択してください</option>
              {categoricalOptions.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">グループ値① (任意)</label>
            <input
              value={groupA}
              onChange={(event) => setGroupA(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              placeholder="例: 男子"
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">グループ値② (任意)</label>
            <input
              value={groupB}
              onChange={(event) => setGroupB(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
              placeholder="例: 女子"
            />
          </div>
        </div>
        {previewGroups.length > 0 && (
          <p className="text-xs text-muted-foreground">サンプル値: {previewGroups.join(", ")}</p>
        )}
        <Button onClick={handleTTest} disabled={tLoading}>
          {tLoading ? "計算中..." : "t検定を実行"}
        </Button>
        {tResult && (
          <div className="space-y-3 rounded-xl border border-primary/30 bg-primary/5 p-4">
            <p className="text-sm text-muted-foreground">
              t統計量: <span className="font-semibold text-foreground">{tResult.t_statistic.toFixed(3)}</span> / p値: {tResult.p_value.toFixed(4)}
            </p>
            <p className="text-sm text-muted-foreground">
              判定: {tResult.significant ? <span className="font-semibold text-primary">有意差あり (p &lt; 0.05)</span> : "有意差なし"}
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg bg-background/60 p-3">
                <p className="text-xs text-muted-foreground">{tResult.group_a}</p>
                <p className="text-lg font-semibold text-foreground">平均: {tResult.mean_a.toFixed(2)}</p>
              </div>
              <div className="rounded-lg bg-background/60 p-3">
                <p className="text-xs text-muted-foreground">{tResult.group_b}</p>
                <p className="text-lg font-semibold text-foreground">平均: {tResult.mean_b.toFixed(2)}</p>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default StatisticsPage;
