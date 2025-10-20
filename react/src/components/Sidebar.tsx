import { useState } from "react";
import { NavLink } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  ArrowLeftRight,
  BarChart3,
  Brain,
  ChartSpline,
  ClipboardList,
  GraduationCap,
  Home,
  LineChart,
  MessageCircle,
  Menu,
  NotebookPen,
  ShieldCheck,
  X,
} from "lucide-react";

import { useAuth } from "../context/AuthContext";
import { cn } from "../lib/utils";
import { Button } from "./ui/button";

interface NavItem {
  to: string;
  label: string;
  description: string;
  icon: typeof Activity;
  roles?: Array<"teacher" | "student">;
}

const NAV_ITEMS: NavItem[] = [
  {
    to: "/dashboard",
    label: "ホーム",
    description: "データの概要とプレビュー",
    icon: Home,
  },
  {
    to: "/statistics",
    label: "統計解析",
    description: "χ²検定やt検定を実施",
    icon: BarChart3,
  },
  {
    to: "/compare",
    label: "データ比較",
    description: "2つのCSVを比較",
    icon: ArrowLeftRight,
  },
  {
    to: "/tasks",
    label: "課題",
    description: "提出状況を確認",
    icon: ClipboardList,
  },
  {
    to: "/regression",
    label: "重回帰分析",
    description: "予測モデルを構築",
    icon: LineChart,
  },
  {
    to: "/factor-analysis",
    label: "因子分析",
    description: "潜在因子を抽出",
    icon: Brain,
  },
  {
    to: "/tasks/create",
    label: "課題作成",
    description: "新しい課題を配布",
    icon: NotebookPen,
    roles: ["teacher"],
  },
  {
    to: "/chat",
    label: "課題チャット",
    description: "課題に関するやり取り",
    icon: MessageCircle,
  },
  {
    to: "/pose",
    label: "骨格推定",
    description: "動画から動作を比較",
    icon: Activity,
    roles: ["student"],
  },
  {
    to: "/admin",
    label: "ユーザー管理",
    description: "承認とパスワード管理",
    icon: ShieldCheck,
    roles: ["teacher"],
  },
];

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const { user } = useAuth();

  const allowedNavItems = NAV_ITEMS.filter((item) => {
    if (!item.roles || !user) {
      return true;
    }
    return item.roles.includes(user.role);
  });

  const renderNav = (onNavigate?: () => void) => (
    <nav className="space-y-1">
      {allowedNavItems.map((item) => {
        const Icon = item.icon;
        return (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all",
                "hover:bg-secondary/80 hover:text-foreground",
                isActive
                  ? "bg-secondary text-foreground shadow-sm"
                  : "text-muted-foreground",
              )
            }
            onClick={() => {
              onNavigate?.();
            }}
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-md bg-secondary group-hover:bg-background">
              <Icon className="h-4 w-4" />
            </span>
            <span className="flex flex-col">
              <span className="font-medium tracking-tight">{item.label}</span>
              <span className="text-xs text-muted-foreground group-hover:text-foreground/80">
                {item.description}
              </span>
            </span>
          </NavLink>
        );
      })}
    </nav>
  );

  return (
    <>
      <div className="sticky top-0 z-40 flex items-center gap-2 border-b bg-background/80 px-4 py-3 backdrop-blur md:hidden">
        <Button variant="ghost" size="icon" onClick={() => setIsOpen((prev) => !prev)}>
          {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </Button>
        <div className="flex items-center gap-2 text-sm font-semibold">
          <GraduationCap className="h-4 w-4" />
          <span>EduRun Studio</span>
        </div>
      </div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 z-30 bg-black/40 md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
          >
            <motion.aside
              className="absolute left-0 top-0 h-full w-72 bg-background p-6 shadow-2xl"
              initial={{ x: -320 }}
              animate={{ x: 0 }}
              exit={{ x: -320 }}
              transition={{ type: "spring", stiffness: 260, damping: 25 }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="mb-8 flex items-center gap-2">
                <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <ChartSpline className="h-5 w-5" />
                </span>
                <div>
                  <p className="font-semibold">EduRun Studio</p>
                  <p className="text-xs text-muted-foreground">学習データ分析</p>
                </div>
              </div>
              {renderNav(() => setIsOpen(false))}
            </motion.aside>
          </motion.div>
        )}
      </AnimatePresence>

      <motion.aside
        className="sticky top-0 hidden h-screen w-72 flex-col overflow-y-auto border-r bg-background/90 px-4 py-6 backdrop-blur md:flex"
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
      >
        <div className="mb-8 flex items-center gap-3 px-2">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <ChartSpline className="h-5 w-5" />
          </span>
          <div>
            <p className="text-sm font-semibold leading-tight">EduRun Studio</p>
            <p className="text-xs text-muted-foreground">走力データプラットフォーム</p>
          </div>
        </div>
        {renderNav()}
      </motion.aside>
    </>
  );
};

export default Sidebar;
