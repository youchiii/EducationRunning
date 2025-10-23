import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import Header from "../components/Header";
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
import { PINK_CHECKBOX_CLASS } from "../constants/styles";
import { SummaryCards } from "../components/regression/SummaryCards";
import { PredVsActualPlot } from "../components/regression/PredVsActualPlot";
import { ResidualsVsFittedPlot } from "../components/regression/ResidualsVsFittedPlot";
import { QQPlot } from "../components/regression/QQPlot";
import { niceNumber } from "../utils/regression";

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
  const [analysisNotes, setAnalysisNotes] = useState("");
  const resultsContentRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!target) {
      return;
    }
    setSelectedFeatures((prev) => prev.filter((feature) => feature !== target));
  }, [target]);

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
      setAnalysisNotes("");
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
                    className={PINK_CHECKBOX_CLASS}
                    disabled={column === target}
                  />
                  <span className={column === target ? "text-muted-foreground" : "text-foreground"}>{column}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <Button
          type="submit"
          disabled={isLoading}
          className="bg-pink-400 text-white hover:bg-pink-300 focus-visible:ring-pink-300"
        >
          {isLoading ? "計算中..." : "重回帰分析を実行"}
        </Button>
      </form>

      {result && (
        <div className="space-y-4 rounded-2xl border border-primary/30 bg-primary/5 p-6 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-lg font-semibold text-foreground">分析結果</h2>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handlePreview}
                className="border-pink-200 bg-pink-50 text-pink-600 hover:bg-pink-100 focus-visible:ring-pink-300"
              >
                👁 プレビュー
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleExportPdf}
                disabled={isExporting}
                className="border-pink-200 bg-pink-50 text-pink-600 hover:bg-pink-100 focus-visible:ring-pink-300"
              >
                {isExporting ? "出力中..." : "📄 PDF出力"}
              </Button>
            </div>
          </div>
          <div ref={resultsContentRef} id="regression-result" className="space-y-4">
            <SummaryCards
              r2={result.r_squared}
              adjustedR2={result.adjusted_r_squared}
              mae={result.mae}
              mape={result.mape}
              durbinWatson={result.dw}
              n={result.n}
            />
            <div className="grid gap-4 md:grid-cols-2">
              <div className="rounded-2xl border border-border/60 bg-background/90 p-4">
                <PredVsActualPlot actual={result.y_true} predicted={result.y_pred} />
              </div>
              <div className="rounded-2xl border border-border/60 bg-background/90 p-4">
                <ResidualsVsFittedPlot
                  predicted={result.y_pred}
                  residuals={result.residuals}
                  stdResiduals={result.std_residuals}
                />
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-background/90 p-4">
              <QQPlot theoretical={result.qq_theoretical} sample={result.qq_sample} />
            </div>
            <div className="space-y-3 rounded-2xl border border-border/60 bg-background/90 p-4">
              <div>
                <p className="text-sm font-semibold text-foreground">分析メモ</p>
                <p className="text-xs text-muted-foreground">PDF出力に含まれる所感・メモを記入できます。</p>
              </div>
              <textarea
                value={analysisNotes}
                onChange={(event) => setAnalysisNotes(event.target.value)}
                rows={6}
                placeholder="分析の気づきや共有事項をここに記入してください。"
                className="min-h-[180px] w-full resize-vertical rounded-lg border border-pink-100 bg-background px-4 py-3 text-sm text-foreground shadow-inner focus:border-pink-300 focus:outline-none focus:ring-2 focus:ring-pink-200"
              />
            </div>
            <CoefficientDetails result={result} />
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

type CoefficientDetailsProps = {
  result: RegressionResponse;
};

const CoefficientDetails = ({ result }: CoefficientDetailsProps) => {
  const rows = useMemo(() => {
    const data = [] as Array<{
      key: string;
      label: string;
      coefficient: number | null;
      stdCoefficient: number | null;
      standardError: number | null;
      pvalue: number | null;
      vif: number | null;
    }>;

    const intercept = result.coefficients["const"];
    if (typeof intercept === "number") {
      data.push({
        key: "const",
        label: "定数項",
        coefficient: intercept,
        stdCoefficient: null,
        standardError: null,
        pvalue: result.pvalues["const"] ?? null,
        vif: null,
      });
    }

    result.features.forEach((feature) => {
      data.push({
        key: feature,
        label: feature,
        coefficient: result.coefficients[feature] ?? null,
        stdCoefficient: result.std_coefficients[feature] ?? null,
        standardError: result.standard_errors[feature] ?? null,
        pvalue: result.pvalues[feature] ?? null,
        vif: result.vif[feature] ?? null,
      });
    });
    return data;
  }, [result]);

  if (rows.length === 0) {
    return null;
  }

  const formatPValue = (value: number | null) => {
    if (value === null || Number.isNaN(value)) {
      return "—";
    }
    if (value < 0.001) {
      return "<0.001";
    }
    return niceNumber(value, 3);
  };

  const hasVif = rows.some((row) => row.vif !== null && Number.isFinite(row.vif ?? NaN));

  return (
    <div className="space-y-3 rounded-2xl border border-border/60 bg-background/90 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-foreground">係数と診断指標</p>
        <p className="text-xs text-muted-foreground">p値&lt;0.05 はハイライト表示</p>
      </div>
      <div className="overflow-auto">
        <table className="min-w-full divide-y divide-border/60 text-sm">
          <thead className="bg-muted/40 text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-3 py-2 text-left">変数</th>
              <th className="px-3 py-2 text-right">係数</th>
              <th className="px-3 py-2 text-right">標準化β</th>
              <th className="px-3 py-2 text-right">標準誤差</th>
              <th className="px-3 py-2 text-right">p値</th>
              {hasVif ? <th className="px-3 py-2 text-right">VIF</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isSignificant = row.pvalue !== null && row.pvalue < 0.05;
              const vifWarning = row.vif !== null && row.vif >= 10;
              const vifCaution = row.vif !== null && row.vif >= 5 && row.vif < 10;
              return (
                <tr key={row.key} className="border-t border-border/40">
                  <td className="px-3 py-2 text-foreground">{row.label}</td>
                  <td className="px-3 py-2 text-right text-foreground">{niceNumber(row.coefficient, 3)}</td>
                  <td className="px-3 py-2 text-right text-foreground">
                    {row.stdCoefficient === null ? "—" : niceNumber(row.stdCoefficient, 3)}
                  </td>
                  <td className="px-3 py-2 text-right text-muted-foreground">
                    {row.standardError === null ? "—" : niceNumber(row.standardError, 3)}
                  </td>
                  <td className={`px-3 py-2 text-right ${isSignificant ? "text-pink-600" : "text-muted-foreground"}`}>
                    {formatPValue(row.pvalue)}
                  </td>
                  {hasVif ? (
                    <td
                      className={`px-3 py-2 text-right ${
                        vifWarning
                          ? "text-red-500"
                          : vifCaution
                            ? "text-amber-500"
                            : "text-muted-foreground"
                      }`}
                    >
                      {row.vif === null ? "—" : niceNumber(row.vif, 2)}
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {hasVif ? (
        <p className="text-xs text-muted-foreground">
          VIF 5以上は多重共線性の注意域、10以上は要対策の目安です。
        </p>
      ) : null}
    </div>
  );
};
