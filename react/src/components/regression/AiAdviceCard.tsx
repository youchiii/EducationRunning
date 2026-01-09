import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Copy, RefreshCcw, Sparkles } from "lucide-react";

import { Button } from "../ui/button";

export type AdviceCopyStatus = "idle" | "success" | "error";

type AiAdviceCardProps = {
  advice: string | null;
  isLoading: boolean;
  onGenerate: () => void;
  onCopy: () => void;
  disabled?: boolean;
  modelUsed?: string | null;
  tokens?: { input: number; output: number } | null;
  copyStatus?: AdviceCopyStatus;
};

const AiAdviceSkeleton = () => (
  <div className="space-y-4 animate-pulse">
    <div className="h-4 w-32 rounded bg-muted/60" />
    <div className="space-y-2 rounded-lg border border-dashed border-border/60 p-4">
      <div className="h-3 w-4/5 rounded bg-muted/40" />
      <div className="h-3 w-3/5 rounded bg-muted/30" />
      <div className="h-3 w-full rounded bg-muted/20" />
    </div>
    <div className="h-3 w-2/5 rounded bg-muted/30" />
  </div>
);

const MarkdownPreview = ({ content }: { content: string }) => (
  <ReactMarkdown
    remarkPlugins={[remarkGfm]}
    components={{
      h2: ({ children }) => (
        <h2 className="mt-4 text-sm font-semibold text-foreground first:mt-0">{children}</h2>
      ),
      p: ({ children }) => <p className="text-sm leading-relaxed text-foreground">{children}</p>,
      li: ({ children }) => <li className="ml-5 list-disc text-sm text-foreground">{children}</li>,
      ul: ({ children }) => <ul className="space-y-1 text-sm text-foreground">{children}</ul>,
      a: ({ children, href }) => (
        <a href={href} target="_blank" rel="noopener" className="text-primary underline">
          {children}
        </a>
      ),
      strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
    }}
  >
    {content}
  </ReactMarkdown>
);

export const AiAdviceCard = ({
  advice,
  isLoading,
  onGenerate,
  onCopy,
  disabled,
  modelUsed,
  tokens,
  copyStatus = "idle",
}: AiAdviceCardProps) => {
  const formattedAdvice = advice?.trim() ?? "";
  const hasAdvice = formattedAdvice.length > 0;
  const buttonLabel = isLoading ? "生成中..." : hasAdvice ? "もう一度生成" : "AI解説";

  return (
    <div className="card card--form space-y-4 rounded-2xl border border-border/60 bg-background/90 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" aria-hidden />
          <h3 className="text-sm font-semibold text-foreground">Geminiによる展望サマリ</h3>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={onGenerate}
            disabled={disabled || isLoading}
            className="bg-primary text-primary-foreground hover:bg-primary/90 focus-visible:ring-primary/50"
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <RefreshCcw className="h-4 w-4 animate-spin" />
                <span>{buttonLabel}</span>
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <Sparkles className="h-4 w-4" />
                <span>{buttonLabel}</span>
              </span>
            )}
          </Button>
          <Button
            variant="outline"
            disabled={!hasAdvice || isLoading}
            onClick={onCopy}
            className="flex items-center gap-2"
          >
            <Copy className="h-4 w-4" />
            コピー
          </Button>
        </div>
      </div>

      {copyStatus === "success" && (
        <p className="text-xs text-emerald-600">クリップボードにコピーしました。</p>
      )}
      {copyStatus === "error" && (
        <p className="text-xs text-destructive">コピーに失敗しました。ブラウザの権限をご確認ください。</p>
      )}

      {isLoading ? (
        <AiAdviceSkeleton />
      ) : hasAdvice ? (
        <div className="space-y-3 text-sm leading-relaxed text-foreground">
          <MarkdownPreview content={formattedAdvice} />
        </div>
      ) : (
        <div className="rounded-lg border border-dashed border-border/50 bg-muted/20 p-4 text-sm text-muted-foreground">
          重回帰のメトリクスと利用者の感想を基に、Gemini が次の検証アイデアを提案します。
        </div>
      )}

      {modelUsed && (
        <p className="text-xs text-muted-foreground">
          使用モデル: {modelUsed}
          {tokens ? `（入力 ${tokens.input} / 出力 ${tokens.output} tokens）` : null}
        </p>
      )}
    </div>
  );
};
