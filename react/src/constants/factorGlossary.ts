export const FACTOR_GLOSSARY = {
  loadings:
    "観測変数と因子の相関の強さを示す係数。絶対値が大きいほど、その変数がその因子を強く表す。各因子に対する寄与の“向きと大きさ”。",
  communalities:
    "その観測変数の分散のうち、抽出した因子で説明できる割合。≈ 各因子負荷量²の合計（0〜1に近いほど因子でよく説明できている）。",
  uniqueness:
    "その観測変数の分散のうち、因子では説明できない部分（固有＋誤差）。= 1 − 共同性（高いほど“その変数固有の要素/誤差”が大きい）。",
  scoreScatter:
    "各サンプル（行）の因子得点を軸（例：F1×F2）にプロットした図。クラスタや外れ値、時間推移のパターンを直感的に把握できる。",
} as const;

export type FactorGlossaryKey = keyof typeof FACTOR_GLOSSARY;
