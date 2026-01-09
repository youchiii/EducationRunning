import type { ChangeEvent } from "react";

type UserNotesInputProps = {
  value: string;
  onChange: (value: string) => void;
  maxLength?: number;
};

export const UserNotesInput = ({ value, onChange, maxLength = 1000 }: UserNotesInputProps) => {
  const remaining = Math.max(0, maxLength - value.length);

  const handleChange = (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange(event.target.value.slice(0, maxLength));
  };

  return (
    <div className="card card--form space-y-3 rounded-2xl border border-border/60 bg-background/90 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-foreground">感想（任意）</p>
          <p className="text-xs text-muted-foreground">
            AIに伝えたい気づきや疑問点があれば記入してください（{maxLength}文字以内）。
          </p>
        </div>
        <span className={`text-xs font-medium ${remaining <= 40 ? "text-destructive" : "text-muted-foreground"}`}>
          残り {remaining} 文字
        </span>
      </div>
      <textarea
        value={value}
        onChange={handleChange}
        maxLength={maxLength}
        rows={4}
        placeholder="例：速度より歩数が効いていそう。高心拍時の外れ値が気になる…"
        className="min-h-[120px] w-full resize-y rounded-lg border border-border/60 bg-background px-3 py-2 text-sm text-foreground shadow-inner focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/40"
      />
    </div>
  );
};
