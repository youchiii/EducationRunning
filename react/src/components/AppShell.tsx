import { useRef, useState } from "react";
import { Outlet } from "react-router-dom";
import { motion } from "framer-motion";
import { UploadCloud } from "lucide-react";

import { useAuth } from "../context/AuthContext";
import { useDataset } from "../context/DatasetContext";
import { Button } from "./ui/button";
import Sidebar from "./Sidebar";

const AppShell = () => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { user, logout } = useAuth();
  const { dataset, uploadDataset, isLoading } = useDataset();
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFileSelect = async (file: File | null) => {
    if (!file) {
      return;
    }
    try {
      setUploadError(null);
      await uploadDataset(file);
    } catch (error) {
      console.error(error);
      setUploadError("CSVの読み込みに失敗しました。ファイル形式を確認してください。");
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const triggerFilePicker = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="flex min-h-screen bg-muted/30">
      <Sidebar />
      <main className="flex min-h-screen flex-1 flex-col">
        <header className="flex flex-col gap-4 border-b bg-background/80 px-4 py-4 backdrop-blur md:flex-row md:items-center md:justify-between md:px-8">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">データセット</p>
            {dataset ? (
              <div className="mt-1">
                <p className="text-lg font-semibold text-foreground">{dataset.original_name}</p>
                <p className="text-sm text-muted-foreground">
                  {dataset.row_count.toLocaleString()} 行・{dataset.column_count} 列
                </p>
              </div>
            ) : (
              <p className="mt-1 text-sm text-muted-foreground">CSVファイルをアップロードすると各種解析が利用できます。</p>
            )}
          </div>
          <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center">
            <div className="flex items-center gap-3 rounded-lg border border-dashed border-muted-foreground/40 px-4 py-2 text-sm text-muted-foreground">
              <UploadCloud className="hidden h-4 w-4 text-primary sm:inline" />
              <div className="leading-tight">
                <p className="font-medium text-foreground">データをアップロード</p>
                <p className="text-xs text-muted-foreground">UTF-8/Shift-JIS 対応 CSV</p>
              </div>
              <Button variant="secondary" size="sm" disabled={isLoading} onClick={triggerFilePicker}>
                {isLoading ? "アップロード中" : "ファイル選択"}
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(event) => handleFileSelect(event.target.files?.[0] ?? null)}
              />
            </div>
            <div className="flex items-center gap-3 rounded-lg bg-secondary/60 px-3 py-2 text-secondary-foreground">
              <div>
                <p className="text-sm font-semibold">{user?.username}</p>
                <p className="text-xs capitalize text-muted-foreground">
                  {user?.role === "teacher" ? "管理者" : "生徒"}
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                className="border-destructive/70 text-destructive hover:bg-destructive/10"
                onClick={logout}
              >
                ログアウト
              </Button>
            </div>
          </div>
        </header>

        <div className="flex-1 px-3 py-6 md:px-8">
          <motion.div
            className="mx-auto flex w-full max-w-6xl flex-col gap-6"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          >
            {uploadError && (
              <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
                {uploadError}
              </div>
            )}
            <Outlet />
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default AppShell;
