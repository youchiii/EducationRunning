import { Button } from "./ui/button";

type ReportToolbarProps = {
  onPreview: () => void;
  onPdf: () => void;
  isPdfLoading?: boolean;
};

const ReportToolbar = ({ onPreview, onPdf, isPdfLoading = false }: ReportToolbarProps) => (
  <div className="sticky top-0 z-50 border-b border-border/60 bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/65">
    <div className="mx-auto flex max-w-[1200px] items-center justify-between px-4 py-3">
      <h2 className="text-sm font-semibold text-foreground sm:text-base">因子分析レポート</h2>
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="min-w-[96px] border-pink-200 bg-pink-50 text-pink-600 hover:bg-pink-100 dark:border-pink-500/60 dark:bg-pink-500/20 dark:text-pink-100"
          onClick={onPreview}
        >
          👁 プレビュー
        </Button>
        <Button
          type="button"
          size="sm"
          className="min-w-[96px] bg-pink-500 text-white hover:bg-pink-400 focus-visible:ring-pink-300 dark:bg-pink-600"
          onClick={onPdf}
          disabled={isPdfLoading}
        >
          {isPdfLoading ? "出力中..." : "📄 PDF"}
        </Button>
      </div>
    </div>
  </div>
);

export default ReportToolbar;
