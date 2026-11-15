import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, setCredentials } from "./api";
import ChatView from "./components/ChatView";
import Sidebar from "./components/Sidebar";
import SignIn from "./components/SignIn";
import type {
  ChatSummary,
  Credentials,
  CurrentUser,
  Fact,
  Message,
  Provider,
  QuestionOptions,
  Repo,
  ServerStatus,
  Source,
} from "./types";

const REPO_KEY = "repomentor.repo";
const CHAT_KEY = "repomentor.chat";
const OPTIONS_KEY = "repomentor.options";
const CREDENTIALS_KEY = "repomentor.credentials";

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
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage can be unavailable (private mode); remembering is optional
  }
}

export interface PendingQuestion {
  chatId: number | null;
  question: string;
  options: QuestionOptions;
  startedAt: number;
  /** What the agent says it is doing, streamed from the server. */
  status: string | null;
  /** Markdown streamed so far (coder, reviewer, tour). */
  text: string;
  /** Facts streamed so far (mentor). */
  facts: { fact: Fact; source: Source }[];
  sources: Source[];
}

/** Loads the server status and handles signing in when the server needs it. */
export default function App() {
  const [status, setStatus] = useState<ServerStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [credentials, setStoredCredentials] = useState<Credentials | null>(() =>
    readStorage<Credentials | null>(CREDENTIALS_KEY, null),
  );
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [signInError, setSignInError] = useState<string | null>(null);

  // The API module sends these with every request
  setCredentials(credentials);

  useEffect(() => {
    api
      .status()
      .then(setStatus)
      .catch((error) => setStatusError(error.message));
  }, []);

  const authRequired = status?.auth_mode === "token";

  useEffect(() => {
    if (!authRequired || !credentials) {
      setUser(null);
      return;
    }
    api.me().then(setUser).catch(() => undefined);
  }, [authRequired, credentials]);

  const signIn = async (value: Credentials) => {
    setCredentials(value);

    try {
      setUser(await api.me());
      writeStorage(CREDENTIALS_KEY, value);
      setStoredCredentials(value);
      setSignInError(null);
    } catch (error) {
      setCredentials(null);
      throw error;
    }
  };

  const signOut = useCallback((reason: string | null = null) => {
    writeStorage(CREDENTIALS_KEY, null);
    writeStorage(REPO_KEY, null);
    writeStorage(CHAT_KEY, null);
    setStoredCredentials(null);
    setUser(null);
    setSignInError(reason);
  }, []);

  if (authRequired && !credentials) {
    return (
      <SignIn
        providers={(status?.providers ?? ["gitlab"]).filter(
          (p): p is "gitlab" | "github" => p !== "local",
        )}
        initialError={signInError}
        onSignIn={signIn}
      />
    );
  }

  return (
    <Workspace
      key={credentials?.token ?? "local"}
      status={status}
      statusError={statusError}
      user={authRequired ? user : null}
      onSignOut={authRequired ? () => signOut() : null}
      onUnauthorized={() =>
        signOut("Your sign-in is no longer valid. Sign in again.")
      }
    />
  );
}

function Workspace({
  status,
  statusError,
  user,
  onSignOut,
  onUnauthorized,
}: {
  status: ServerStatus | null;
  statusError: string | null;
  user: CurrentUser | null;
  onSignOut: (() => void) | null;
  onUnauthorized: () => void;
}) {
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
  const [error, setError] = useState<string | null>(statusError);
  // Options saved by older versions lack newer fields such as `agent`
  const [options, setOptions] = useState<QuestionOptions>(() => ({
    ...DEFAULT_OPTIONS,
    ...readStorage<Partial<QuestionOptions>>(OPTIONS_KEY, {}),
  }));
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => setError(statusError), [statusError]);

  // The chat the user is looking at, readable from async callbacks
  const activeChatRef = useRef(activeChatId);
  activeChatRef.current = activeChatId;

  // Changing the active chat reloads its messages. A chat created by
  // sending its first question is empty, so that load is skipped once.
  const skipNextLoad = useRef<number | null>(null);

  const selectedRepo = repos.find((repo) => repo.id === selectedRepoId) ?? null;

  const showError = useCallback(
    (value: unknown) => {
      if (value instanceof ApiError && value.status === 401) {
        onUnauthorized();
        return;
      }
      setError(value instanceof Error ? value.message : String(value));
    },
    [onUnauthorized],
  );

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
    refreshRepos();
  }, [refreshRepos]);

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

  const addRepo = async (projectPath: string, provider: Provider | null) => {
    const repo = await api.addRepo(projectPath, provider);
    setRepos((list) =>
      [...list, repo].sort((a, b) =>
        a.project_path.localeCompare(b.project_path),
      ),
    );
    setSelectedRepoId(repo.id);
    setActiveChatId(null);
  };

  const reindexRepo = async (repoId: number, full: boolean) => {
    try {
      const repo = await api.reindexRepo(repoId, full);
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
      status: null,
      text: "",
      facts: [],
      sources: [],
    });

    const update = (change: (current: PendingQuestion) => PendingQuestion) =>
      setPending((current) => (current ? change(current) : current));

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
        update((current) => ({ ...current, chatId: chat.id }));

        if (activeChatRef.current === null) {
          // Switch to it without reloading: the messages arrive below
          activeChatRef.current = chat.id;
          skipNextLoad.current = chat.id;
          setActiveChatId(chat.id);
        }
      }

      const saved: Message[] = [];

      await api.askStream(chatId, question, askOptions, (event) => {
        switch (event.type) {
          case "user":
          case "message":
            saved.push(event.message);
            break;
          case "status":
            update((current) => ({ ...current, status: event.label }));
            break;
          case "sources":
            update((current) => ({ ...current, sources: event.sources }));
            break;
          case "delta":
            update((current) => ({ ...current, text: current.text + event.text }));
            break;
          case "fact":
            update((current) => ({
              ...current,
              facts: [...current.facts, { fact: event.fact, source: event.source }],
            }));
            break;
        }
      });

      if (activeChatRef.current === chatId) {
        setMessages((list) => [...list, ...saved]);
      }

      refreshChats(repoId);
    } catch (value) {
      showError(value);
    } finally {
      setPending(null);
    }
  };

  const replaceMessage = (message: Message) =>
    setMessages((list) => list.map((m) => (m.id === message.id ? message : m)));

  const activeChat = chats.find((chat) => chat.id === activeChatId) ?? null;
  const pendingHere =
    pending !== null && pending.chatId === activeChatId ? pending : null;

  return (
    <div className={`app ${sidebarOpen ? "sidebar-open" : ""}`}>
      <Sidebar
        status={status}
        user={user}
        onSignOut={onSignOut}
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
        onMessageChange={replaceMessage}
        onError={showError}
        onNewChat={() => selectChat(null)}
        onOpenSidebar={() => setSidebarOpen(true)}
        error={error}
        onDismissError={() => setError(null)}
      />
    </div>
  );
}
