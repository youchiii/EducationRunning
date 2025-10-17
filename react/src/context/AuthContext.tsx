import {
  type PropsWithChildren,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  fetchCurrentUser,
  login as loginRequest,
  signup as signupRequest,
  type AuthenticatedUser,
  type LoginPayload,
  type SignupPayload,
  setAuthToken,
} from "../services/api";

const TOKEN_STORAGE_KEY = "rundata_auth_token";

type AuthContextValue = {
  user: AuthenticatedUser | null;
  token: string | null;
  isLoading: boolean;
  login: (payload: LoginPayload) => Promise<void>;
  signup: (payload: SignupPayload) => Promise<string>;
  logout: () => void;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export const AuthProvider = ({ children }: PropsWithChildren) => {
  const [user, setUser] = useState<AuthenticatedUser | null>(null);
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(TOKEN_STORAGE_KEY));
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (token) {
      setAuthToken(token);
      fetchCurrentUser()
        .then((profile) => {
          setUser(profile);
        })
        .catch(() => {
          setUser(null);
          setToken(null);
          setAuthToken(null);
          localStorage.removeItem(TOKEN_STORAGE_KEY);
        })
        .finally(() => setIsLoading(false));
    } else {
      setAuthToken(null);
      setIsLoading(false);
    }
  }, [token]);

  const login = useCallback(async (payload: LoginPayload) => {
    const response = await loginRequest(payload);
    setToken(response.access_token);
    setAuthToken(response.access_token);
    localStorage.setItem(TOKEN_STORAGE_KEY, response.access_token);
    setUser(response.user);
  }, []);

  const signup = useCallback(async (payload: SignupPayload) => {
    const response = await signupRequest(payload);
    return response.message;
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setToken(null);
    setAuthToken(null);
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  }, []);

  const refresh = useCallback(async () => {
    if (!token) {
      setUser(null);
      return;
    }
    try {
      const profile = await fetchCurrentUser();
      setUser(profile);
    } catch (error) {
      logout();
      throw error;
    }
  }, [token, logout]);

  const value = useMemo<AuthContextValue>(
    () => ({ user, token, isLoading, login, logout, signup, refresh }),
    [user, token, isLoading, login, logout, signup, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
};
