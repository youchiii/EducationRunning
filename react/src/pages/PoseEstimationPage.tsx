import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent } from "react";
import { motion } from "framer-motion";
import { UploadCloud, Video } from "lucide-react";
import Header from "../components/Header";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import VideoPreview from "../components/VideoPreview";
import DTWChart from "../components/DTWChart";
import {
  API_BASE_URL,
  fetchResults,
  triggerAnalysis,
  uploadVideos,
  type ResultsResponse,
} from "../services/api";
import { loadSessionId, persistSessionId } from "../utils/session";
import { cn } from "../lib/utils";
import { useAuth } from "../context/AuthContext";

const motionCard = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

const PoseEstimationPage = () => {
  const { user } = useAuth();
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [comparisonFile, setComparisonFile] = useState<File | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);
  const [comparisonPreview, setComparisonPreview] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(loadSessionId());
  const [results, setResults] = useState<ResultsResponse | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (referencePreview) URL.revokeObjectURL(referencePreview);
      if (comparisonPreview) URL.revokeObjectURL(comparisonPreview);
    };
  }, [referencePreview, comparisonPreview]);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    const loadLatestResults = async () => {
      try {
        setStatusMessage("最新の解析結果を読み込んでいます...");
        const response = await fetchResults(sessionId);
        setResults(response);
        setStatusMessage(null);
      } catch (fetchError) {
        console.error(fetchError);
        setStatusMessage(null);
      }
    };

    void loadLatestResults();
  }, [sessionId]);

  if (user?.role !== "student") {
    return (
      <div className="space-y-4">
        <Header title="骨格推定" subtitle="生徒アカウント専用の機能です。" />
        <p className="rounded-lg border border-dashed border-border/60 bg-background/80 p-6 text-sm text-muted-foreground">
          生徒アカウントでログインすると、動画の骨格推定とDTWによる比較結果を確認できます。
        </p>
      </div>
    );
  }

  const handleFileSelect = (
    event: ChangeEvent<HTMLInputElement>,
    type: "reference" | "comparison",
  ) => {
    const file = event.target.files?.[0] ?? null;
    if (type === "reference") {
      setReferenceFile(file);
      if (referencePreview) URL.revokeObjectURL(referencePreview);
      setReferencePreview(file ? URL.createObjectURL(file) : null);
    } else {
      setComparisonFile(file);
      if (comparisonPreview) URL.revokeObjectURL(comparisonPreview);
      setComparisonPreview(file ? URL.createObjectURL(file) : null);
    }
  };

  const runAnalysis = async () => {
    if (!referenceFile || !comparisonFile) {
      setErrorMessage("参照動画と比較動画の両方を選択してください。");
      return;
    }

    try {
      setIsProcessing(true);
      setErrorMessage(null);
      setStatusMessage("動画をアップロードしています...");

      const formData = new FormData();
      formData.append("reference_video", referenceFile);
      formData.append("comparison_video", comparisonFile);
      const uploadResponse = await uploadVideos(formData);

      persistSessionId(uploadResponse.session_id);
      setSessionId(uploadResponse.session_id);

      setStatusMessage("解析を開始しています...");
      const analysisResponse = await triggerAnalysis({
        session_id: uploadResponse.session_id,
      });

      setResults(analysisResponse);
      setStatusMessage("解析結果を取得しました。");
      setTimeout(() => setStatusMessage(null), 2000);
    } catch (processError) {
      console.error(processError);
      setErrorMessage("解析に失敗しました。バックエンドが起動しているか確認してください。");
      setStatusMessage(null);
    } finally {
      setIsProcessing(false);
    }
  };

  const distanceSeries = useMemo(() => {
    if (!results?.metrics.path) {
      return [];
    }
    const { query, reference } = results.metrics.path;
    return query.map((value, index) => Math.abs(value - (reference[index] ?? value)));
  }, [results]);

  const makeAbsoluteUrl = (path?: string | null) => {
    if (!path) {
      return undefined;
    }
    if (/^https?:/.test(path)) {
      return path;
    }
    return `${API_BASE_URL}${path}`;
  };

  return (
    <div className="space-y-8">
      <Header
        title="骨格推定"
        subtitle="動画から骨格を推定し、動きの類似度を分析します。"
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <motion.div {...motionCard} transition={{ delay: 0.05, duration: 0.35 }}>
          <Card className="h-full overflow-hidden border-dashed border-muted-foreground/40">
            <CardHeader className="space-y-1">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Video className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wider">参照動画</span>
              </div>
              <CardTitle className="text-lg">参照動画を選択</CardTitle>
              <CardDescription>Streamlitアプリと同じく、左側に参照動画を配置します。</CardDescription>
            </CardHeader>
            <CardContent>
              <label
                className={cn(
                  "group flex h-44 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-muted bg-muted/40 text-center text-sm text-muted-foreground transition",
                  "hover:border-primary/60 hover:bg-primary/5 hover:text-foreground",
                )}
              >
                <UploadCloud className="h-7 w-7 text-muted-foreground transition group-hover:text-primary" />
                <div>
                  <p className="font-medium">ファイルを選択</p>
                  <p className="text-xs text-muted-foreground">
                    MP4などの動画ファイルをドラッグ&ドロップまたはクリックで選択
                  </p>
                </div>
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(event) => handleFileSelect(event, "reference")}
                />
              </label>
              {referencePreview && (
                <div className="mt-4 overflow-hidden rounded-lg border">
                  <video src={referencePreview} controls className="w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div {...motionCard} transition={{ delay: 0.1, duration: 0.35 }}>
          <Card className="h-full overflow-hidden border-dashed border-muted-foreground/40">
            <CardHeader className="space-y-1">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Video className="h-4 w-4" />
                <span className="text-xs uppercase tracking-wider">比較動画</span>
              </div>
              <CardTitle className="text-lg">比較動画を選択</CardTitle>
              <CardDescription>右側に比較する動画を配置します。</CardDescription>
            </CardHeader>
            <CardContent>
              <label
                className={cn(
                  "group flex h-44 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed border-muted bg-muted/40 text-center text-sm text-muted-foreground transition",
                  "hover:border-primary/60 hover:bg-primary/5 hover:text-foreground",
                )}
              >
                <UploadCloud className="h-7 w-7 text-muted-foreground transition group-hover:text-primary" />
                <div>
                  <p className="font-medium">ファイルを選択</p>
                  <p className="text-xs text-muted-foreground">
                    参照動画と比較する動画をアップロードしてください
                  </p>
                </div>
                <input
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(event) => handleFileSelect(event, "comparison")}
                />
              </label>
              {comparisonPreview && (
                <div className="mt-4 overflow-hidden rounded-lg border">
                  <video src={comparisonPreview} controls className="w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <motion.div {...motionCard} transition={{ delay: 0.15, duration: 0.35 }}>
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">解析フロー</CardTitle>
            <CardDescription>
              Streamlitと同様に、動画をアップロードして「解析開始」を押すとDTW解析が実行されます。
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-muted-foreground">
            <p>1. 参照動画と比較動画をアップロードします。</p>
            <p>2. 「解析開始」を押すとサーバー側で骨格推定とDTW計算が行われます。</p>
            <p>3. 結果が下部に表示され、プレビュー動画も再生できます。</p>
          </CardContent>
          <CardFooter className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-sm font-medium text-muted-foreground">
              {statusMessage ? statusMessage : "準備が整ったら解析を開始してください。"}
            </div>
            <Button onClick={runAnalysis} disabled={isProcessing} className="shadow-lg">
              {isProcessing ? "解析中..." : "解析開始"}
            </Button>
          </CardFooter>
          {errorMessage && (
            <div className="px-6 pb-4 text-sm text-destructive">{errorMessage}</div>
          )}
        </Card>
      </motion.div>

      {results && (
        <motion.section {...motionCard} transition={{ delay: 0.2, duration: 0.35 }} className="space-y-6">
          <Tabs defaultValue="summary" className="w-full">
            <TabsList>
              <TabsTrigger value="summary">解析結果</TabsTrigger>
              <TabsTrigger value="visual">プレビュー</TabsTrigger>
            </TabsList>
            <TabsContent value="summary" className="bg-background p-6">
              <div className="grid gap-4 lg:grid-cols-3">
                <Card className="border-transparent bg-gradient-to-br from-primary/10 via-background to-background">
                  <CardHeader>
                    <CardDescription>類似度</CardDescription>
                    <CardTitle className="text-3xl font-semibold">
                      {results.metrics.similarity_percentage.toFixed(2)}%
                    </CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader>
                    <CardDescription>DTW距離</CardDescription>
                    <CardTitle className="text-3xl font-semibold">
                      {results.metrics.dtw_distance.toFixed(2)}
                    </CardTitle>
                  </CardHeader>
                </Card>
                <Card>
                  <CardHeader>
                    <CardDescription>最終更新</CardDescription>
                    <CardTitle className="text-base font-semibold">
                      {results.updated_at
                        ? new Date(results.updated_at).toLocaleString()
                        : "--"}
                    </CardTitle>
                  </CardHeader>
                </Card>
              </div>

              <div className="mt-6">
                <DTWChart
                  referenceSeries={results.metrics.path.reference}
                  comparisonSeries={results.metrics.path.query}
                  distanceSeries={distanceSeries}
                />
              </div>
            </TabsContent>

            <TabsContent value="visual" className="bg-background p-6">
              <div className="grid gap-6 md:grid-cols-2">
                <VideoPreview
                  src={makeAbsoluteUrl(results.preview_videos.reference)}
                  title="参照動画プレビュー"
                  placeholder="解析後に骨格プレビューが表示されます。"
                />
                <VideoPreview
                  src={makeAbsoluteUrl(results.preview_videos.comparison)}
                  title="比較動画プレビュー"
                  placeholder="解析後に骨格プレビューが表示されます。"
                />
              </div>
            </TabsContent>
          </Tabs>
        </motion.section>
      )}
    </div>
  );
};

export default PoseEstimationPage;
