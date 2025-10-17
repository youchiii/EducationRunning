import { useRef } from "react";
import { FileText, UploadCloud, X } from "lucide-react";

import { Button } from "./ui/button";

interface FileUploaderProps {
  file: File | null;
  onChange: (file: File | null) => void;
  accept?: string;
  label?: string;
}

const formatBytes = (size: number) => {
  if (!size) {
    return "";
  }
  if (size < 1024) {
    return `${size}B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)}KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(1)}MB`;
};

const FileUploader = ({ file, onChange, accept, label = "ファイルをアップロード" }: FileUploaderProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const openPicker = () => {
    inputRef.current?.click();
  };

  return (
    <div className="space-y-3">
      <Button type="button" variant="outline" className="flex items-center gap-2" onClick={openPicker}>
        <UploadCloud className="h-4 w-4" />
        {label}
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(event) => {
          const selected = event.target.files?.[0] ?? null;
          onChange(selected);
        }}
      />
      {file ? (
        <div className="flex items-center justify-between rounded-2xl border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-900 shadow-sm">
          <div className="flex items-center gap-3">
            <FileText className="h-5 w-5" />
            <div>
              <p className="font-medium">{file.name}</p>
              <p className="text-xs text-blue-700/70">{formatBytes(file.size)}</p>
            </div>
          </div>
          <button
            type="button"
            className="rounded-full p-1 text-blue-500 transition hover:bg-white"
            onClick={() => onChange(null)}
            aria-label="アップロードを取り消す"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      ) : (
        <p className="text-xs text-gray-500">PDFやZip、Officeファイルなど提出したい資料を添付してください。</p>
      )}
    </div>
  );
};

export default FileUploader;
