import axios, { type AxiosHeaders, type AxiosRequestHeaders } from "axios";

const DEFAULT_API_URL = "http://localhost:8000";
export const API_BASE_URL = import.meta.env.VITE_API_URL ?? DEFAULT_API_URL;

let authToken: string | null = null;

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

apiClient.interceptors.request.use((config) => {
  if (authToken) {
    const headers = (config.headers ?? {}) as AxiosRequestHeaders & Record<string, string>;
    if (typeof (headers as unknown as AxiosHeaders).set === "function") {
      (headers as unknown as AxiosHeaders).set("Authorization", `Bearer ${authToken}`);
    } else {
      headers.Authorization = `Bearer ${authToken}`;
    }
    config.headers = headers;
  }
  return config;
});

export const setAuthToken = (token: string | null) => {
  authToken = token;
};

// Auth

export type AuthenticatedUser = {
  id: number;
  username: string;
  role: "teacher" | "student";
  status: string;
  created_at?: string;
};

export type LoginPayload = {
  username: string;
  password: string;
};

export type LoginResponse = {
  access_token: string;
  token_type: string;
  user: AuthenticatedUser;
};

export type SignupPayload = {
  username: string;
  password: string;
};

export const login = async (payload: LoginPayload) => {
  const { data } = await apiClient.post<LoginResponse>("/auth/login", payload);
  return data;
};

export const signup = async (payload: SignupPayload) => {
  const { data } = await apiClient.post<{ message: string }>("/auth/signup", payload);
  return data;
};

export const fetchCurrentUser = async () => {
  const { data } = await apiClient.get<AuthenticatedUser>("/auth/me");
  return data;
};

// Dataset ingestion & statistics

export type DatasetPreview = {
  dataset_id: string;
  original_name: string;
  row_count: number;
  column_count: number;
  preview: Record<string, unknown>[];
  numeric_columns: string[];
  categorical_columns: string[];
};

export type DatasetStats = {
  dataset_id: string;
  row_count: number;
  column_count: number;
  numeric_columns: string[];
  basic_statistics: Record<string, {
    mean: number | null;
    median: number | null;
    mode: number | null;
    variance: number | null;
    std_dev: number | null;
  }>;
  correlation_matrix: Record<string, Record<string, number | null>>;
};

export type SeriesResponse = {
  dataset_id: string;
  column: string;
  values: Array<number | null>;
};

export type ChiSquareResponse = {
  chi2: number;
  p_value: number;
  dof: number;
  significant: boolean;
  contingency_table: Record<string, unknown>[];
};

export type TTestResponse = {
  t_statistic: number;
  p_value: number;
  group_a: string;
  group_b: string;
  significant: boolean;
  mean_a: number;
  mean_b: number;
};

export const uploadDataset = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post<DatasetPreview>("/data/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const fetchDatasetStats = async (datasetId: string) => {
  const { data } = await apiClient.get<DatasetStats>(`/data/${datasetId}/stats`);
  return data;
};

export const fetchColumnSeries = async (datasetId: string, column: string) => {
  const { data } = await apiClient.get<SeriesResponse>(`/data/${datasetId}/column/${encodeURIComponent(column)}/series`);
  return data;
};

export const runChiSquare = async (datasetId: string, columnA: string, columnB: string) => {
  const { data } = await apiClient.post<ChiSquareResponse>(`/data/${datasetId}/tests/chi2`, {
    column_a: columnA,
    column_b: columnB,
  });
  return data;
};

export const runTTest = async (
  datasetId: string,
  numericColumn: string,
  groupColumn: string,
  groupValues?: string[],
) => {
  const { data } = await apiClient.post<TTestResponse>(`/data/${datasetId}/tests/t`, {
    numeric_column: numericColumn,
    group_column: groupColumn,
    group_values: groupValues,
  });
  return data;
};

// Analysis

export type RegressionResponse = {
  dataset_id: string;
  target: string;
  features: string[];
  coefficients: Record<string, number | null>;
  std_coefficients: Record<string, number | null>;
  standard_errors: Record<string, number | null>;
  pvalues: Record<string, number | null>;
  intercept: number | null;
  r_squared: number | null;
  adjusted_r_squared: number | null;
  mae: number | null;
  mape: number | null;
  dw: number | null;
  y_true: Array<number | null>;
  y_pred: Array<number | null>;
  residuals: Array<number | null>;
  std_residuals: Array<number | null>;
  qq_theoretical: Array<number | null>;
  qq_sample: Array<number | null>;
  vif: Record<string, number | null>;
  n: number;
  index: string[];
};

export type RegressionAdviceRequest = {
  session_id: string;
  metrics: {
    r2?: number | null;
    adj_r2?: number | null;
    mae?: number | null;
    mape?: number | null;
    dw?: number | null;
    n: number;
  };
  coefficients: Record<string, number>;
  std_coefficients?: Record<string, number>;
  pvalues?: Record<string, number>;
  vif?: Record<string, number>;
  residuals_summary?: {
    mean?: number;
    std?: number;
    skew?: number;
    kurt?: number;
    outliers_gt2?: number;
  };
  notes?: string;
  target_name: string;
  feature_names: string[];
};

export type RegressionAdviceResponse = {
  advice: string | {
    summary?: string;
    insights?: string[];
    risks?: string[];
    next_actions?: string[];
  };
  model_used: string;
  tokens?: {
    input: number;
    output: number;
  };
};

export type FactorLoadingRow = {
  variable: string;
  [key: `factor_${number}`]: number;
};

export type FactorScorePreviewRow = {
  index: number;
  [key: `factor_${number}`]: number;
};

export type FactorScoreRow = {
  row_index: number;
  [key: `factor_${number}`]: number;
};

export type FactorUploadResponse = {
  session_id: string;
  columns: string[];
  n_rows: number;
};

export type FactorScreeResponse = {
  eigenvalues: number[];
  explained_variance_ratio: number[];
};

export type FactorRunResponse = {
  loadings: Record<string, Record<string, number>>;
  communalities: Record<string, number>;
  uniqueness: Record<string, number>;
  factor_scores: Array<Record<string, number>>;
  n_rows: number;
};

export type FactorRegressionColumnTarget = {
  type: "column";
  name: string;
};

export type FactorRegressionArrayTarget = {
  type: "array";
  values: number[];
};

export type FactorRegressionTarget = FactorRegressionColumnTarget | FactorRegressionArrayTarget;

export type FactorRegressionRequest = {
  session_id: string;
  target: FactorRegressionTarget;
  factors?: string[];
  standardize_target?: boolean;
  sample_limit?: number;
};

export type FactorRegressionResponse = {
  coefficients: Record<string, number>;
  std_coefficients: Record<string, number>;
  pvalues: Record<string, number>;
  r2: number;
  adj_r2: number;
  dw: number;
  vif: Record<string, number>;
  fitted: number[];
  residuals: number[];
  indices: number[];
  qq_theoretical: number[];
  qq_sample: number[];
  used_factors: string[];
  n: number;
};

export type FactorAutoNFactorsResponse = {
  recommended_n: number;
  by_rule: {
    pa: number;
    kaiser: number;
    elbow: number;
    cum: number;
  };
  cumvar: number[];
  eigenvalues: number[];
  pa_threshold: number[];
  kaiser: number;
  rationale: string;
  n_samples: number;
  n_vars: number;
  target_cumvar: number;
  pa_percentile: number;
};

export type FactorAutoNFactorsRequest = {
  session_id: string;
  target_cumvar?: number;
  pa_iter?: number;
  pa_percentile?: number;
  max_factors?: number | null;
};

export type FactorAutoExplainResponse = {
  explanation: string;
  recommended_n: number;
};

export type FactorAutoExplainPayload = {
  n_vars: number;
  n_samples: number;
  eigenvalues: number[];
  pa_threshold?: number[] | null;
  cumvar?: number[] | null;
  by_rule: Record<string, number>;
  recommended_n: number;
  notes?: string;
};

export const runRegression = async (
  datasetId: string,
  target: string,
  features: string[],
  testSize?: number,
) => {
  const { data } = await apiClient.post<RegressionResponse>("/analysis/regression", {
    dataset_id: datasetId,
    target,
    features,
    test_size: testSize,
  });
  return data;
};

export const fetchRegressionAdvice = async (payload: RegressionAdviceRequest) => {
  const { data } = await apiClient.post<RegressionAdviceResponse>("/ai/regression-advice", payload);
  return data;
};

export const uploadFactorDataset = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post<FactorUploadResponse>("/fa/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const createFactorSessionFromDataset = async (datasetId: string) => {
  const { data } = await apiClient.post<FactorUploadResponse>("/fa/from-dataset", {
    dataset_id: datasetId,
  });
  return data;
};

export const fetchFactorScree = async (sessionId: string) => {
  const { data } = await apiClient.get<FactorScreeResponse>("/fa/scree", {
    params: { session_id: sessionId },
  });
  return data;
};

export const runFactorAnalysisSession = async (
  sessionId: string,
  nFactors: number,
  columns?: string[],
) => {
  const { data } = await apiClient.post<FactorRunResponse>("/fa/run", {
    session_id: sessionId,
    n_factors: nFactors,
    columns,
  });
  return data;
};

export const runFactorRegression = async (payload: FactorRegressionRequest) => {
  const { data } = await apiClient.post<FactorRegressionResponse>("/fa/regression", payload);
  return data;
};

export const fetchAutoFactorRecommendation = async (payload: FactorAutoNFactorsRequest) => {
  const { data } = await apiClient.post<FactorAutoNFactorsResponse>("/fa/auto_n_factors", payload);
  return data;
};

export const fetchAutoFactorExplanation = async (payload: FactorAutoExplainPayload) => {
  const { data } = await apiClient.post<FactorAutoExplainResponse>("/fa/auto_n_factors_explain", payload);
  return data;
};

// Chat

export type ConversationLatestMessage = {
  id: number;
  text: string;
  timestamp: string;
};

export type ConversationSummary = {
  conversation_id: number;
  partner_id: number;
  partner_name: string;
  latest_message: ConversationLatestMessage | null;
  unread_count: number;
};

export type ChatMessage = {
  id: number;
  conversation_id: number;
  sender_id: number;
  receiver_id: number;
  text: string;
  reply_to_id: number | null;
  timestamp: string;
  is_read: boolean;
};

export type SendChatMessagePayload = {
  sender_id: number;
  receiver_id: number;
  text: string;
  reply_to_id?: number | null;
};

export type SendChatMessageResponse = {
  conversation_id: number;
  message: ChatMessage;
};

export type MarkReadResponse = {
  conversation_id: number;
  updated_count: number;
};

export const fetchChatConversations = async () => {
  const { data } = await apiClient.get<ConversationSummary[]>("/chat/conversations");
  return data;
};

export const fetchChatMessages = async (conversationId: number) => {
  const { data } = await apiClient.get<ChatMessage[]>(`/chat/messages/${conversationId}`);
  return data;
};

export const sendChatMessage = async (payload: SendChatMessagePayload) => {
  const { data } = await apiClient.post<SendChatMessageResponse>("/chat/send", payload);
  return data;
};

export const markConversationRead = async (conversationId: number) => {
  const { data } = await apiClient.patch<MarkReadResponse>(`/chat/read/${conversationId}`);
  return data;
};

// Tasks

export type TaskStudent = {
  id: number;
  username: string;
};

export type TaskSummary = {
  id: number;
  title: string;
  description: string | null;
  teacher_id: number;
  student_ids: number[];
  deadline: string | null;
  file_url: string | null;
  original_filename: string | null;
  created_at: string;
  student_status?: "pending" | "submitted" | "late" | "overdue";
  submitted_at?: string | null;
  is_overdue?: boolean | null;
  submitted_count?: number | null;
  total_assignees?: number | null;
};

export type TaskDetail = TaskSummary & {
  teacher_name?: string | null;
  students: TaskStudent[];
};

export type TaskSubmission = {
  id: number | null;
  student_id: number;
  student_name: string;
  file_url: string | null;
  status: "submitted" | "late" | "pending" | "overdue";
  submitted_at: string | null;
};

export type CreateTaskPayload = {
  title: string;
  description?: string;
  deadline?: string | null;
  targetStudents: number[];
  file?: File | null;
};

export const fetchTaskStudents = async () => {
  const { data } = await apiClient.get<TaskStudent[]>("/tasks/students");
  return data;
};

export const fetchTasks = async () => {
  const { data } = await apiClient.get<TaskSummary[]>("/tasks");
  return data;
};

export const createTask = async ({ title, description, deadline, targetStudents, file }: CreateTaskPayload) => {
  const formData = new FormData();
  formData.append("title", title);
  if (description) {
    formData.append("description", description);
  }
  if (deadline) {
    formData.append("deadline", deadline);
  }
  targetStudents.forEach((studentId) => {
    formData.append("target_students", String(studentId));
  });
  if (file) {
    formData.append("file", file);
  }

  const { data } = await apiClient.post<TaskDetail>("/tasks", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const fetchTaskDetail = async (taskId: number) => {
  const { data } = await apiClient.get<TaskDetail>(`/tasks/${taskId}`);
  return data;
};

export const submitTask = async (taskId: number, file: File) => {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await apiClient.post<TaskSubmission>(`/tasks/${taskId}/submit`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
};

export const fetchTaskSubmissions = async (taskId: number) => {
  const { data } = await apiClient.get<TaskSubmission[]>(`/tasks/${taskId}/submissions`);
  return data;
};

// Admin

export type UserSummary = {
  id: number;
  username: string;
  role: string;
  status: string;
  created_at: string;
};

export const fetchPendingUsers = async () => {
  const { data } = await apiClient.get<UserSummary[]>("/admin/pending");
  return data;
};

export const fetchAllUsers = async () => {
  const { data } = await apiClient.get<UserSummary[]>("/admin/users");
  return data;
};

export const updateUserStatus = async (userId: number, status: "active" | "rejected") => {
  await apiClient.post(`/admin/users/${userId}/status`, { status });
};

export const resetUserPassword = async (userId: number, newPassword: string) => {
  await apiClient.post(`/admin/users/${userId}/reset-password`, { new_password: newPassword });
};

export const deleteUser = async (userId: number) => {
  await apiClient.delete(`/admin/users/${userId}`);
};

// Pose estimation

export type UploadResponse = {
  session_id: string;
  reference_video: string;
  comparison_video: string;
  uploaded_at: string;
};

export interface AnalyzePayload {
  session_id: string;
  model_complexity?: number;
  min_detection_confidence?: number;
  min_tracking_confidence?: number;
}

export type AnalysisSettings = Required<AnalyzePayload>;

export type DTWPath = {
  query: number[];
  reference: number[];
};

export interface Metrics {
  dtw_distance: number;
  similarity_percentage: number;
  path: DTWPath;
}

export interface ResultsResponse {
  session_id: string;
  metrics: Metrics;
  analysis_settings: AnalyzePayload;
  preview_videos: {
    reference: string;
    comparison: string;
  };
  downloads: {
    metrics: string;
    reference_landmarks: string;
    comparison_landmarks: string;
  };
  source_videos: {
    reference: string;
    comparison: string;
  };
  original_filenames: {
    reference?: string | null;
    comparison?: string | null;
  };
  uploaded_at?: string;
  updated_at?: string;
}

export const uploadVideos = async (formData: FormData) => {
  const response = await apiClient.post<UploadResponse>("/pose/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const triggerAnalysis = async (payload: AnalyzePayload) => {
  const response = await apiClient.post<ResultsResponse>("/pose/analyze", payload);
  return response.data;
};

export const fetchResults = async (sessionId: string) => {
  const response = await apiClient.get<ResultsResponse>("/pose/results", {
    params: { session_id: sessionId },
  });
  return response.data;
};

export type DatasetUploadResult = DatasetPreview;
