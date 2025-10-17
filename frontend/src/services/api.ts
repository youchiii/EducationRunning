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
  r_squared: number;
  mse: number;
  intercept: number;
  coefficients: Array<{ feature: string; coefficient: number }>;
  equation: string;
  predictions: Array<{ actual: number; predicted: number }>;
};

export type FactorAnalysisResponse = {
  dataset_id: string;
  columns: string[];
  n_components: number;
  factor_loadings: Array<Record<string, number | string>>;
  factor_scores_preview: Array<Record<string, number>>;
  explained_variance_ratio: number[];
  cumulative_variance_ratio: number[];
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

export const runFactorAnalysis = async (
  datasetId: string,
  columns: string[],
  nComponents: number,
) => {
  const { data } = await apiClient.post<FactorAnalysisResponse>("/analysis/factor", {
    dataset_id: datasetId,
    columns,
    n_components: nComponents,
  });
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
