import { useRef, useState } from "react";
import type { FormEvent } from "react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import Header from "../components/Header";
import RegressionScatter from "../components/RegressionScatter";
import { Button } from "../components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";
import { useDataset } from "../context/DatasetContext";
import { runRegression, type RegressionResponse } from "../services/api";

const RegressionPage = () => {
  const { dataset, stats } = useDataset();
  const [target, setTarget] = useState("");
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [result, setResult] = useState<RegressionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const resultsContentRef = useRef<HTMLDivElement | null>(null);

  if (!dataset || !stats) {
    return (
      <div className="space-y-4">
        <Header title="重回帰分析" subtitle="まずはCSVデータをアップロードしてください。" />
        <p className="rounded-lg border border-dashed border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
          数値列が読み込まれると目的変数と説明変数を選択できます。
        </p>
      </div>
    );
  }

  const handleFeatureToggle = (feature: string) => {
    setSelectedFeatures((prev) => (prev.includes(feature) ? prev.filter((item) => item !== feature) : [...prev, feature]));
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!target || selectedFeatures.length === 0) {
      setError("目的変数と説明変数を選択してください。");
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const response = await runRegression(dataset.dataset_id, target, selectedFeatures);
      setResult(response);
    } catch (requestError) {
      console.error(requestError);
      setError("重回帰分析の実行に失敗しました。データ数や欠損値を確認してください。");
    } finally {
      setIsLoading(false);
    }
  };

  const createCanvasImage = async () => {
    if (!resultsContentRef.current) {
      return null;
    }
    const canvas = await html2canvas(resultsContentRef.current, {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
    });
    return { dataUrl: canvas.toDataURL("image/png"), canvas };
  };

  const handleExportPdf = async () => {
    if (!resultsContentRef.current) {
      return;
    }
    setIsExporting(true);
    try {
      const capture = await createCanvasImage();
      if (!capture) {
        return;
      }
      const { dataUrl, canvas } = capture;
      const pdf = new jsPDF("p", "pt", "a4");
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = pageWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let heightLeft = imgHeight;
      let position = 0;

      pdf.addImage(dataUrl, "PNG", 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      while (heightLeft > 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(dataUrl, "PNG", 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      const datasetName = dataset?.original_name?.replace(/[^a-zA-Z0-9-_]/g, "_") ?? "dataset";
      const timestamp = new Date().toISOString().split("T")[0];
      pdf.save(`${datasetName}-regression-${timestamp}.pdf`);
    } catch (exportError) {
      console.error("Failed to export regression report", exportError);
    } finally {
      setIsExporting(false);
    }
  };

  const handlePreview = async () => {
    try {
      const capture = await createCanvasImage();
      if (capture) {
        setPreviewImage(capture.dataUrl);
        setIsPreviewOpen(true);
      }
    } catch (previewError) {
      console.error("Failed to render preview", previewError);
    }
  };

  return (
    <div className="space-y-6">
      <Header
        title="重回帰分析"
        subtitle="複数の説明変数から目的変数を予測するモデルを構築します。"
      />
      {error && <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{error}</div>}

      <form className="space-y-4 rounded-2xl border border-border/60 bg-background/90 p-6 shadow-sm" onSubmit={handleSubmit}>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">目的変数</label>
            <select
              value={target}
              onChange={(event) => setTarget(event.target.value)}
              className="w-full rounded-lg border border-border/60 bg-background px-3 py-2 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
            >
              <option value="">選択してください</option>
              {stats.numeric_columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">説明変数</label>
            <div className="max-h-48 space-y-1 overflow-auto rounded-lg border border-border/60 bg-background p-3 text-sm">
              {stats.numeric_columns.map((column) => (
                <label key={column} className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={selectedFeatures.includes(column)}
                    onChange={() => handleFeatureToggle(column)}
                    className="h-4 w-4 rounded border-border accent-primary"
                    disabled={column === target}
                  />
                  <span className={column === target ? "text-muted-foreground" : "text-foreground"}>{column}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <Button type="submit" disabled={isLoading}>
          {isLoading ? "計算中..." : "重回帰分析を実行"}
        </Button>
      </form>

      {result && (
        <div className="space-y-4 rounded-2xl border border-primary/30 bg-primary/5 p-6 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-lg font-semibold text-foreground">分析結果</h2>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handlePreview}>
                👁 プレビュー
              </Button>
              <Button variant="outline" size="sm" onClick={handleExportPdf} disabled={isExporting}>
                {isExporting ? "出力中..." : "📄 PDF出力"}
              </Button>
            </div>
          </div>
          <div ref={resultsContentRef} id="regression-result" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-lg bg-background/60 p-4">
              <p className="text-xs text-muted-foreground">決定係数 (R²)</p>
              <p className="text-2xl font-semibold text-foreground">{result.r_squared.toFixed(3)}</p>
            </div>
            <div className="rounded-lg bg-background/60 p-4">
              <p className="text-xs text-muted-foreground">平均二乗誤差 (MSE)</p>
              <p className="text-2xl font-semibold text-foreground">{result.mse.toFixed(3)}</p>
            </div>
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">回帰式</p>
            <p className="mt-1 rounded-md bg-background/80 p-3 text-sm text-muted-foreground">{result.equation}</p>
          </div>
          <div className="space-y-3 rounded-xl border border-border/60 bg-background/80 p-4">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-foreground">実測値 vs 予測値</p>
              <p className="text-xs text-muted-foreground">破線は y = x を示します</p>
            </div>
            <RegressionScatter points={result.predictions} />
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-3">
              <p className="text-sm font-medium text-foreground">係数</p>
              <div className="overflow-hidden rounded-lg border border-border/60">
                <table className="min-w-full divide-y divide-border/60 text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-3 py-2 text-left text-muted-foreground">変数</th>
                      <th className="px-3 py-2 text-right text-muted-foreground">係数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.coefficients.map((item) => (
                      <tr key={item.feature} className="border-t border-border/40">
                        <td className="px-3 py-2 text-foreground">{item.feature}</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{item.coefficient.toFixed(3)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="space-y-3">
              <p className="text-sm font-medium text-foreground">実測値 vs 予測値</p>
              <div className="overflow-auto rounded-lg border border-border/60">
                <table className="min-w-full divide-y divide-border/60 text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="px-3 py-2 text-right text-muted-foreground">実測値</th>
                      <th className="px-3 py-2 text-right text-muted-foreground">予測値</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.predictions.map((pair, index) => (
                      <tr key={index} className="border-t border-border/40">
                        <td className="px-3 py-2 text-right text-foreground">{pair.actual.toFixed(2)}</td>
                        <td className="px-3 py-2 text-right text-muted-foreground">{pair.predicted.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          </div>
          <Dialog
            open={isPreviewOpen}
            onOpenChange={(open) => {
              setIsPreviewOpen(open);
              if (!open) {
                setPreviewImage(null);
              }
            }}
          >
            <DialogContent className="max-h-[90vh] max-w-5xl overflow-hidden">
              <DialogHeader>
                <DialogTitle>PDFプレビュー</DialogTitle>
                <DialogDescription>表示内容をそのままPDFに保存できます。</DialogDescription>
              </DialogHeader>
              <div className="flex justify-center overflow-y-auto rounded-xl bg-white p-6 shadow-inner">
                {previewImage ? (
                  <img
                    src={previewImage}
                    alt="Regression preview"
                    className="w-full max-w-[794px] rounded-lg border border-gray-200 shadow-lg transition"
                    style={{ aspectRatio: "1 / 1.414" }}
                  />
                ) : (
                  <p className="text-sm text-muted-foreground">プレビューを生成できませんでした。</p>
                )}
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setIsPreviewOpen(false)}>
                  閉じる
                </Button>
                <Button onClick={async () => {
                  await handleExportPdf();
                  setIsPreviewOpen(false);
                }}>
                  PDFとして保存
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      )}
    </div>
  );
};

export default RegressionPage;
