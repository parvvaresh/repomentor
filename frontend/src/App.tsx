import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import ChatView from "./components/ChatView";
import Sidebar from "./components/Sidebar";
import type {
  ChatSummary,
  Message,
  QuestionOptions,
  Repo,
  ServerStatus,
} from "./types";

const REPO_KEY = "repomentor.repo";
const CHAT_KEY = "repomentor.chat";
const OPTIONS_KEY = "repomentor.options";

const DEFAULT_OPTIONS: QuestionOptions = {
  top_k: 5,
  retrieval_only: false,
  agent: "mentor",
};

function readStorage<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : (JSON.parse(value) as T);
  } catch {
    return fallback;
  }
}

function writeStorage(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage can be unavailable (private mode); remembering is optional
  }
}

export interface PendingQuestion {
  chatId: number | null;
  question: string;
  options: QuestionOptions;
  startedAt: number;
}

export default function App() {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [repos, setRepos] = useState<Repo[]>([]);
  const [reposLoaded, setReposLoaded] = useState(false);
  const [selectedRepoId, setSelectedRepoId] = useState<number | null>(() =>
    readStorage<number | null>(REPO_KEY, null),
  );
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(() =>
    readStorage<number | null>(CHAT_KEY, null),
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadingChat, setLoadingChat] = useState(false);
  const [pending, setPending] = useState<PendingQuestion | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Options saved by older versions lack newer fields such as `agent`
  const [options, setOptions] = useState<QuestionOptions>(() => ({
    ...DEFAULT_OPTIONS,
    ...readStorage<Partial<QuestionOptions>>(OPTIONS_KEY, {}),
  }));
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // The chat the user is looking at, readable from async callbacks
  const activeChatRef = useRef(activeChatId);
  activeChatRef.current = activeChatId;

  // Changing the active chat reloads its messages. A chat created by
  // sending its first question is empty, so that load is skipped once.
  const skipNextLoad = useRef<number | null>(null);

  const selectedRepo = repos.find((repo) => repo.id === selectedRepoId) ?? null;

  const showError = useCallback((value: unknown) => {
    setError(value instanceof Error ? value.message : String(value));
  }, []);

  const refreshRepos = useCallback(async () => {
    try {
      const list = await api.listRepos();
      setRepos(list);
      return list;
    } catch (value) {
      showError(value);
      return null;
    } finally {
      setReposLoaded(true);
    }
  }, [showError]);

  const refreshChats = useCallback(
    async (repoId: number) => {
      try {
        setChats(await api.listChats(repoId));
      } catch (value) {
        showError(value);
      }
    },
    [showError],
  );

  // Initial load
  useEffect(() => {
    api.status().then(setStatus).catch(showError);
    refreshRepos();
  }, [refreshRepos, showError]);

  // Pick a repository once the list is known
  useEffect(() => {
    if (!reposLoaded) return;

    if (selectedRepoId !== null && repos.some((r) => r.id === selectedRepoId)) {
      return;
    }

    const fallback = repos.find((r) => r.status === "ready") ?? repos[0];
    setSelectedRepoId(fallback ? fallback.id : null);
  }, [repos, reposLoaded, selectedRepoId]);

  // Poll while a repository is being indexed
  const anyIndexing = repos.some((repo) => repo.status === "indexing");

  useEffect(() => {
    if (!anyIndexing) return;

    const timer = window.setInterval(refreshRepos, 2000);
    return () => window.clearInterval(timer);
  }, [anyIndexing, refreshRepos]);

  // Load the chat list of the selected repository
  useEffect(() => {
    writeStorage(REPO_KEY, selectedRepoId);

    if (selectedRepoId === null) {
      setChats([]);
      return;
    }

    let cancelled = false;

    api
      .listChats(selectedRepoId)
      .then((list) => {
        if (cancelled) return;

        setChats(list);
        setActiveChatId((current) =>
          current !== null && list.some((chat) => chat.id === current)
            ? current
            : null,
        );
      })
      .catch(showError);

    return () => {
      cancelled = true;
    };
  }, [selectedRepoId, showError]);

  // Load the messages of the active chat
  useEffect(() => {
    writeStorage(CHAT_KEY, activeChatId);

    if (activeChatId === null) {
      setMessages([]);
      return;
    }

    if (skipNextLoad.current === activeChatId) {
      skipNextLoad.current = null;
      return;
    }

    let cancelled = false;
    setLoadingChat(true);

    api
      .getChat(activeChatId)
      .then((chat) => {
        if (!cancelled) setMessages(chat.messages);
      })
      .catch((value) => {
        if (cancelled) return;
        showError(value);
        setActiveChatId(null);
      })
      .finally(() => {
        if (!cancelled) setLoadingChat(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeChatId, showError]);

  useEffect(() => writeStorage(OPTIONS_KEY, options), [options]);

  // Fall back to the default agent if the saved one no longer exists
  useEffect(() => {
    if (!status || status.agents.some((a) => a.id === options.agent)) return;
    setOptions((current) => ({ ...current, agent: status.default_agent }));
  }, [status, options.agent]);

  const selectRepo = (repoId: number) => {
    if (repoId === selectedRepoId) return;
    setSelectedRepoId(repoId);
    setActiveChatId(null);
    setSidebarOpen(false);
  };

  const selectChat = (chatId: number | null) => {
    setActiveChatId(chatId);
    setSidebarOpen(false);
  };

  const addRepo = async (projectPath: string) => {
    const repo = await api.addRepo(projectPath);
    setRepos((list) =>
      [...list, repo].sort((a, b) =>
        a.project_path.localeCompare(b.project_path),
      ),
    );
    setSelectedRepoId(repo.id);
    setActiveChatId(null);
  };

  const reindexRepo = async (repoId: number) => {
    try {
      const repo = await api.reindexRepo(repoId);
      setRepos((list) => list.map((r) => (r.id === repo.id ? repo : r)));
    } catch (value) {
      showError(value);
    }
  };

  const deleteRepo = async (repoId: number) => {
    try {
      await api.deleteRepo(repoId);
      setRepos((list) => list.filter((r) => r.id !== repoId));

      if (repoId === selectedRepoId) {
        setSelectedRepoId(null);
        setActiveChatId(null);
      }
    } catch (value) {
      showError(value);
    }
  };

  const renameChat = async (chatId: number, title: string) => {
    try {
      await api.renameChat(chatId, title);
      setChats((list) =>
        list.map((chat) => (chat.id === chatId ? { ...chat, title } : chat)),
      );
    } catch (value) {
      showError(value);
    }
  };

  const deleteChat = async (chatId: number) => {
    try {
      await api.deleteChat(chatId);
      setChats((list) => list.filter((chat) => chat.id !== chatId));

      if (chatId === activeChatId) setActiveChatId(null);
    } catch (value) {
      showError(value);
    }
  };

  const ask = async (question: string) => {
    if (!selectedRepo || pending) return;

    const repoId = selectedRepo.id;
    const startChatId = activeChatId;
    const askOptions = { ...options };

    setError(null);
    setPending({
      chatId: startChatId,
      question,
      options: askOptions,
      startedAt: Date.now(),
    });

    try {
      // A chat is only created when its first question is sent
      let chatId = startChatId;

      if (chatId === null) {
        const created = await api.createChat(repoId);
        // Show the title the server will give it while the answer loads
        const firstLine = question.split("\n")[0];
        const chat = {
          ...created,
          title:
            firstLine.length > 60
              ? `${firstLine.slice(0, 59).trimEnd()}…`
              : firstLine,
        };
        chatId = chat.id;
        setChats((list) => [chat, ...list]);
        setPending((current) => current && { ...current, chatId: chat.id });

        if (activeChatRef.current === null) {
          // Switch to it without reloading: the messages arrive below
          activeChatRef.current = chat.id;
          skipNextLoad.current = chat.id;
          setActiveChatId(chat.id);
        }
      }

      const response = await api.ask(chatId, question, askOptions);

      if (activeChatRef.current === chatId) {
        setMessages((list) => [...list, ...response.messages]);
      }

      refreshChats(repoId);
    } catch (value) {
      showError(value);
    } finally {
      setPending(null);
    }
  };

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? null;
  const pendingHere =
    pending !== null && pending.chatId === activeChatId ? pending : null;

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <Sidebar
        status={status}
        repos={repos}
        selectedRepo={selectedRepo}
        chats={chats}
        activeChatId={activeChatId}
        busyChatId={pending?.chatId ?? null}
        onSelectRepo={selectRepo}
        onAddRepo={addRepo}
        onReindexRepo={reindexRepo}
        onDeleteRepo={deleteRepo}
        onSelectChat={selectChat}
        onRenameChat={renameChat}
        onDeleteChat={deleteChat}
        onClose={() => setSidebarOpen(false)}
      />
      <div
        className="sidebar-backdrop"
        onClick={() => setSidebarOpen(false)}
        aria-hidden="true"
      />
      <ChatView
        repo={selectedRepo}
        reposLoaded={reposLoaded}
        status={status}
        chatTitle={activeChat?.title ?? null}
        messages={messages}
        loading={loadingChat}
        pending={pendingHere}
        busyElsewhere={pending !== null && pendingHere === null}
        options={options}
        onOptionsChange={setOptions}
        onAsk={ask}
        onNewChat={() => selectChat(null)}
        onOpenSidebar={() => setSidebarOpen(true)}
        error={error}
        onDismissError={() => setError(null)}
      />
    </div>
  );
}
