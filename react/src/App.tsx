import { type ReactElement, useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import AppShell from "./components/AppShell";
import SplashScreen from "./components/SplashScreen";
import { useAuth } from "./context/AuthContext";
import DashboardPage from "./pages/DashboardPage";
import FactorAnalysisPage from "./pages/FactorAnalysisPage";
import LoginPage from "./pages/LoginPage";
import PoseEstimationPage from "./pages/PoseEstimationPage";
import RegressionPage from "./pages/RegressionPage";
import SignupPage from "./pages/SignupPage";
import StatisticsPage from "./pages/StatisticsPage";
import ChatListPage from "./pages/ChatListPage";
import ChatRoomPage from "./pages/ChatRoomPage";
import TaskListPage from "./pages/TaskListPage";
import TaskDetailPage from "./pages/TaskDetailPage";
import TaskCreatePage from "./pages/TaskCreatePage";
import AdminPage from "./pages/AdminPage";
import DataComparisonPage from "./pages/DataComparisonPage";

const useSplashDelay = (active: boolean, minimumDuration = 2000) => {
  const [visible, setVisible] = useState(active);

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout> | undefined;
    if (active) {
      setVisible(true);
    } else {
      timeout = setTimeout(() => {
        setVisible(false);
      }, minimumDuration);
    }

    return () => {
      if (timeout) {
        clearTimeout(timeout);
      }
    };
  }, [active, minimumDuration]);

  return visible;
};

const ProtectedLayout = () => {
  const { user, isLoading } = useAuth();
  const location = useLocation();
  const showSplash = useSplashDelay(isLoading);

  if (showSplash) {
    return <SplashScreen />;
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return (
    <AppShell />
  );
};

const PublicOnly = ({ children }: { children: ReactElement }) => {
  const { user, isLoading } = useAuth();
  const showSplash = useSplashDelay(isLoading);

  if (showSplash) {
    return <SplashScreen />;
  }

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

const App = () => {
  return (
    <Routes>
      <Route
        path="/login"
        element={
          <PublicOnly>
            <LoginPage />
          </PublicOnly>
        }
      />
      <Route
        path="/signup"
        element={
          <PublicOnly>
            <SignupPage />
          </PublicOnly>
        }
      />

      <Route element={<ProtectedLayout />}>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/statistics" element={<StatisticsPage />} />
        <Route path="/compare" element={<DataComparisonPage />} />
        <Route path="/regression" element={<RegressionPage />} />
        <Route path="/factor-analysis" element={<FactorAnalysisPage />} />
        <Route path="/factor" element={<Navigate to="/factor-analysis" replace />} />
        <Route path="/pose" element={<PoseEstimationPage />} />
        <Route path="/chat" element={<ChatListPage />} />
        <Route path="/chat/:conversationId" element={<ChatRoomPage />} />
        <Route path="/tasks">
          <Route index element={<TaskListPage />} />
          <Route path="create" element={<TaskCreatePage />} />
          <Route path=":taskId" element={<TaskDetailPage />} />
        </Route>
        <Route path="/admin" element={<AdminPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
};

export default App;
