import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { PendingQuestion } from "../App";
import type {
  Message,
  QuestionOptions,
  Repo,
  ServerStatus,
} from "../types";
import AssistantMessage from "./AssistantMessage";
import Icon from "./Icon";

interface Props {
  repo: Repo | null;
  reposLoaded: boolean;
  status: ServerStatus | null;
  chatTitle: string | null;
  messages: Message[];
  loading: boolean;
  pending: PendingQuestion | null;
  busyElsewhere: boolean;
  options: QuestionOptions;
  onOptionsChange: (options: QuestionOptions) => void;
  onAsk: (question: string) => void;
  onNewChat: () => void;
  onOpenSidebar: () => void;
  error: string | null;
  onDismissError: () => void;
}

const EXAMPLE_QUESTIONS = [
  "How does user authentication work?",
  "How are user passwords handled?",
  "Where are the database models defined?",
  "How is the application deployed?",
];

export default function ChatView(props: Props) {
  const { repo, messages, pending, loading } = props;
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view
  useEffect(() => {
    const element = scrollRef.current;
    if (element) element.scrollTop = element.scrollHeight;
  }, [messages, pending, loading]);

  const ready = repo?.status === "ready";
  const empty = messages.length === 0 && !pending && !loading;

  return (
    <main className="chat">
      <header className="chat-header">
        <button
          className="icon-button menu-button"
          onClick={props.onOpenSidebar}
          aria-label="Open sidebar"
        >
          <Icon name="menu" />
        </button>
        <div className="chat-header-text">
          <span className="chat-header-title" dir="auto">
            {props.chatTitle ?? "New chat"}
          </span>
          {repo && <span className="chat-header-repo">{repo.project_path}</span>}
        </div>
        {messages.length > 0 && (
          <button className="ghost-button" onClick={props.onNewChat}>
            <Icon name="plus" />
            <span className="hide-narrow">New chat</span>
          </button>
        )}
      </header>

      <div className="messages" ref={scrollRef}>
        <div className="messages-inner">
          {loading && (
            <div className="center-note">
              <span className="spinner" />
            </div>
          )}

          {empty && (
            <EmptyState
              repo={repo}
              reposLoaded={props.reposLoaded}
              onAsk={props.onAsk}
              canAsk={ready && !props.busyElsewhere}
            />
          )}

          {!loading &&
            messages.map((message) =>
              message.role === "user" ? (
                <UserMessage key={message.id} message={message} />
              ) : (
                <AssistantMessage key={message.id} message={message} />
              ),
            )}

          {pending && (
            <>
              <UserBubble
                text={pending.question}
                options={pending.options}
              />
              <PendingAnswer pending={pending} />
            </>
          )}
        </div>
      </div>

      {props.error && (
        <div className="error-banner" role="alert">
          <Icon name="alert" />
          <span>{props.error}</span>
          <button
            className="icon-button"
            onClick={props.onDismissError}
            aria-label="Dismiss"
          >
            <Icon name="close" />
          </button>
        </div>
      )}

      <Composer
        disabled={!ready || pending !== null || props.busyElsewhere}
        placeholder={composerPlaceholder(repo, props.busyElsewhere)}
        status={props.status}
        options={props.options}
        onOptionsChange={props.onOptionsChange}
        onSubmit={props.onAsk}
      />
    </main>
  );
}

function composerPlaceholder(repo: Repo | null, busyElsewhere: boolean) {
  if (!repo) return "Add a repository to start asking questions";
  if (repo.status === "indexing") return "Waiting for the repository to finish indexing…";
  if (repo.status === "error") return "Re-index this repository to ask questions";
  if (busyElsewhere) return "Waiting for the answer in another chat…";
  return `Ask about ${repo.project_path}…`;
}

function EmptyState({
  repo,
  reposLoaded,
  canAsk,
  onAsk,
}: {
  repo: Repo | null;
  reposLoaded: boolean;
  canAsk: boolean;
  onAsk: (question: string) => void;
}) {
  if (!reposLoaded) return null;

  if (!repo) {
    return (
      <div className="empty-state">
        <h1>Understand any codebase</h1>
        <p className="muted">
          Add a GitLab repository from the sidebar. RepoMentor indexes its
          code and answers your questions with citations to exact files and
          lines.
        </p>
      </div>
    );
  }

  return (
    <div className="empty-state">
      <h1>Ask about {repo.project_path}</h1>
      <p className="muted">
        Answers are built only from retrieved source code, and every fact
        links to the file and lines it came from.
      </p>
      <div className="examples">
        {EXAMPLE_QUESTIONS.map((question) => (
          <button
            key={question}
            className="example"
            disabled={!canAsk}
            onClick={() => onAsk(question)}
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}

function UserMessage({ message }: { message: Message }) {
  const options =
    message.result && "top_k" in message.result ? message.result : null;

  return <UserBubble text={message.content} options={options} />;
}

function UserBubble({
  text,
  options,
}: {
  text: string;
  options: QuestionOptions | null;
}) {
  return (
    <div className="message user-message">
      <div className="user-bubble" dir="auto">
        {text}
      </div>
      {options?.retrieval_only && (
        <span className="message-tag">Retrieval only · top {options.top_k}</span>
      )}
    </div>
  );
}

function PendingAnswer({ pending }: { pending: PendingQuestion }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - pending.startedAt) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [pending.startedAt]);

  const step = pending.options.retrieval_only
    ? "Retrieving sources…"
    : elapsed < 2
      ? "Retrieving sources…"
      : "Reading sources and extracting facts…";

  return (
    <div className="message assistant-message">
      <div className="pending">
        <span className="spinner" />
        <span>{step}</span>
        {elapsed >= 3 && <span className="muted">{elapsed}s</span>}
      </div>
    </div>
  );
}

function Composer({
  disabled,
  placeholder,
  status,
  options,
  onOptionsChange,
  onSubmit,
}: {
  disabled: boolean;
  placeholder: string;
  status: ServerStatus | null;
  options: QuestionOptions;
  onOptionsChange: (options: QuestionOptions) => void;
  onSubmit: (question: string) => void;
}) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to a limit
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 200)}px`;
  }, [value]);

  const submit = () => {
    const question = value.trim();
    if (!question || disabled) return;
    onSubmit(question);
    setValue("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  const maxTopK = status?.max_top_k ?? 20;
  const ollamaOffline = status !== null && !status.ollama_available;

  return (
    <div className="composer-wrap">
      <div className={`composer ${disabled ? "is-disabled" : ""}`}>
        <textarea
          ref={textareaRef}
          rows={1}
          dir="auto"
          value={value}
          placeholder={placeholder}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Question"
        />
        <div className="composer-bar">
          <label className="option" title="Number of source files to retrieve">
            <span>Sources</span>
            <input
              type="number"
              min={1}
              max={maxTopK}
              value={options.top_k}
              onChange={(event) => {
                const topK = Math.max(
                  1,
                  Math.min(maxTopK, Number(event.target.value) || 1),
                );
                onOptionsChange({ ...options, top_k: topK });
              }}
            />
          </label>
          <label
            className="option"
            title="Show retrieved sources without calling the language model"
          >
            <input
              type="checkbox"
              checked={options.retrieval_only}
              onChange={(event) =>
                onOptionsChange({
                  ...options,
                  retrieval_only: event.target.checked,
                })
              }
            />
            <span>Retrieval only</span>
          </label>
          {ollamaOffline && !options.retrieval_only && (
            <span className="option warning" title="Start Ollama to get answers">
              Ollama offline
            </span>
          )}
          <span className="spacer" />
          <button
            className="send-button"
            onClick={submit}
            disabled={disabled || !value.trim()}
            aria-label="Send question"
          >
            <Icon name="send" />
          </button>
        </div>
      </div>
      <p className="composer-hint">
        Enter to send · Shift+Enter for a new line
      </p>
    </div>
  );
}
