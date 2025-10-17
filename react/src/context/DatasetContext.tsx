import {
  type PropsWithChildren,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import {
  fetchColumnSeries,
  fetchDatasetStats,
  type DatasetPreview,
  type DatasetStats,
  uploadDataset as uploadDatasetRequest,
} from "../services/api";

export type SeriesCache = Record<string, Array<number | null>>;

type DatasetContextValue = {
  dataset: DatasetPreview | null;
  stats: DatasetStats | null;
  isLoading: boolean;
  uploadDataset: (file: File) => Promise<void>;
  refreshStats: () => Promise<void>;
  clearDataset: () => void;
  getSeries: (column: string) => Promise<Array<number | null>>;
};

const DatasetContext = createContext<DatasetContextValue | undefined>(undefined);

export const DatasetProvider = ({ children }: PropsWithChildren) => {
  const [dataset, setDataset] = useState<DatasetPreview | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [seriesCache, setSeriesCache] = useState<SeriesCache>({});

  const uploadDataset = useCallback(async (file: File) => {
    setIsLoading(true);
    try {
      const preview = await uploadDatasetRequest(file);
      setDataset(preview);
      const latestStats = await fetchDatasetStats(preview.dataset_id);
      setStats(latestStats);
      setSeriesCache({});
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshStats = useCallback(async () => {
    if (!dataset) {
      return;
    }
    setIsLoading(true);
    try {
      const latest = await fetchDatasetStats(dataset.dataset_id);
      setStats(latest);
    } finally {
      setIsLoading(false);
    }
  }, [dataset]);

  const clearDataset = useCallback(() => {
    setDataset(null);
    setStats(null);
    setSeriesCache({});
  }, []);

  const getSeries = useCallback(
    async (column: string) => {
      if (!dataset) {
        throw new Error("No dataset loaded");
      }
      if (seriesCache[column]) {
        return seriesCache[column];
      }
      const response = await fetchColumnSeries(dataset.dataset_id, column);
      setSeriesCache((cache) => ({ ...cache, [column]: response.values }));
      return response.values;
    },
    [dataset, seriesCache],
  );

  const value = useMemo<DatasetContextValue>(
    () => ({ dataset, stats, isLoading, uploadDataset, refreshStats, clearDataset, getSeries }),
    [dataset, stats, isLoading, uploadDataset, refreshStats, clearDataset, getSeries],
  );

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
};

export const useDataset = () => {
  const context = useContext(DatasetContext);
  if (!context) {
    throw new Error("useDataset must be used within DatasetProvider");
  }
  return context;
};
