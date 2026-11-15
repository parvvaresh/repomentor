import { useState, type FormEvent } from "react";
import type {
  ChatSummary,
  CurrentUser,
  Provider,
  Repo,
  ServerStatus,
} from "../types";
import { formatRelativeTime } from "../time";
import { repoName } from "../types";
import Icon from "./Icon";

interface Props {
  status: ServerStatus | null;
  user: CurrentUser | null;
  onSignOut: (() => void) | null;
  repos: Repo[];
  selectedRepo: Repo | null;
  chats: ChatSummary[];
  activeChatId: number | null;
  busyChatId: number | null;
  onSelectRepo: (repoId: number) => void;
  onAddRepo: (projectPath: string, provider: Provider | null) => Promise<void>;
  onReindexRepo: (repoId: number, full: boolean) => void;
  onDeleteRepo: (repoId: number) => void;
  onSelectChat: (chatId: number | null) => void;
  onRenameChat: (chatId: number, title: string) => void;
  onDeleteChat: (chatId: number) => void;
  onClose: () => void;
}

export default function Sidebar(props: Props) {
  const { status, repos, selectedRepo, chats, activeChatId } = props;
  const [adding, setAdding] = useState(false);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">
          <span className="brand-mark">R</span>
          <span>RepoMentor</span>
        </div>
        <button
          className="icon-button sidebar-close"
          onClick={props.onClose}
          aria-label="Close sidebar"
        >
          <Icon name="close" />
        </button>
      </div>

      <section className="sidebar-section">
        <div className="section-label">
          <span>Repository</span>
          <button
            className="text-button"
            onClick={() => setAdding((value) => !value)}
          >
            {adding ? "Cancel" : "+ Add"}
          </button>
        </div>

        {adding && (
          <AddRepoForm
            providers={status?.providers ?? ["gitlab"]}
            gitlabConfigured={status?.gitlab_configured ?? true}
            onAdd={async (path, provider) => {
              await props.onAddRepo(path, provider);
              setAdding(false);
            }}
          />
        )}

        {repos.length > 0 ? (
          <div className="select-wrap">
            <select
              value={selectedRepo?.id ?? ""}
              onChange={(event) => props.onSelectRepo(Number(event.target.value))}
              aria-label="Repository"
            >
              {repos.map((repo) => (
                <option key={repo.id} value={repo.id}>
                  {repo.provider === "github" ? "GitHub · " : ""}
                  {repoName(repo)}
                  {repo.status === "indexing" ? " (indexing…)" : ""}
                  {repo.status === "error" ? " (error)" : ""}
                </option>
              ))}
            </select>
            <Icon name="chevron" className="select-chevron" />
          </div>
        ) : (
          !adding && (
            <p className="muted small">
              No repositories yet. Add a GitLab or GitHub project to get started.
            </p>
          )
        )}

        {selectedRepo && (
          <RepoDetails
            repo={selectedRepo}
            onReindex={(full) => props.onReindexRepo(selectedRepo.id, full)}
            onDelete={() => props.onDeleteRepo(selectedRepo.id)}
          />
        )}
      </section>

      <button
        className="new-chat-button"
        onClick={() => props.onSelectChat(null)}
        disabled={!selectedRepo}
      >
        <Icon name="plus" />
        New chat
      </button>

      <nav className="chat-list" aria-label="Chat history">
        {selectedRepo && chats.length === 0 && (
          <p className="muted small chat-list-empty">
            Your conversations about this repository will appear here.
          </p>
        )}
        {chats.map((chat) => (
          <ChatItem
            key={chat.id}
            chat={chat}
            active={chat.id === activeChatId}
            busy={chat.id === props.busyChatId}
            onSelect={() => props.onSelectChat(chat.id)}
            onRename={(title) => props.onRenameChat(chat.id, title)}
            onDelete={() => props.onDeleteChat(chat.id)}
          />
        ))}
      </nav>

      <footer className="sidebar-footer">
        {props.user && (
          <div className="user-row">
            <span className="user-name" title={props.user.id}>
              {props.user.name || props.user.username}
            </span>
            {props.onSignOut && (
              <button className="text-button" onClick={props.onSignOut}>
                Sign out
              </button>
            )}
          </div>
        )}
        {status ? (
          <span
            className="model-status"
            title={
              status.ollama_available
                ? "Ollama is running"
                : "Ollama is not reachable. Only retrieval-only questions will work."
            }
          >
            <span
              className={`dot ${status.ollama_available ? "dot-ok" : "dot-bad"}`}
            />
            {status.model}
            {!status.ollama_available && " · offline"}
          </span>
        ) : (
          <span className="muted small">Connecting…</span>
        )}
      </footer>
    </aside>
  );
}

const PROVIDER_LABELS: Record<Provider, string> = {
  gitlab: "GitLab",
  github: "GitHub",
  local: "Local",
};

function AddRepoForm({
  providers,
  gitlabConfigured,
  onAdd,
}: {
  providers: Provider[];
  gitlabConfigured: boolean;
  onAdd: (projectPath: string, provider: Provider | null) => Promise<void>;
}) {
  const [value, setValue] = useState("");
  const [provider, setProvider] = useState<Provider>(providers[0] ?? "gitlab");
  // A pasted URL names its host; the server works it out from the URL
  const isUrl = value.includes("://") || value.startsWith("git@");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();

    if (!value.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      await onAdd(value.trim(), isUrl ? null : provider);
    } catch (value) {
      setError(value instanceof Error ? value.message : String(value));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form className="add-repo" onSubmit={submit}>
      {providers.length > 1 && !isUrl && (
        <div className="segmented small" role="radiogroup" aria-label="Host">
          {providers.map((option) => (
            <button
              type="button"
              key={option}
              role="radio"
              aria-checked={provider === option}
              className={provider === option ? "is-active" : ""}
              onClick={() => setProvider(option)}
            >
              {PROVIDER_LABELS[option]}
            </button>
          ))}
        </div>
      )}
      <input
        autoFocus
        value={value}
        onChange={(event) => setValue(event.target.value)}
        placeholder={
          provider === "github" ? "owner/repo or GitHub URL" : "group/project or GitLab URL"
        }
        aria-label="Project path or URL"
        disabled={submitting}
      />
      <button className="primary-button" disabled={submitting || !value.trim()}>
        {submitting ? "Adding…" : "Index repository"}
      </button>
      {!gitlabConfigured && provider === "gitlab" && !isUrl && (
        <p className="form-hint warning">
          GITLAB_TOKEN or GITLAB_BASE_URL is missing from .env on the server.
        </p>
      )}
      {error && <p className="form-hint error-text">{error}</p>}
    </form>
  );
}

function RepoDetails({
  repo,
  onReindex,
  onDelete,
}: {
  repo: Repo;
  onReindex: (full: boolean) => void;
  onDelete: () => void;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <div className={`repo-card repo-${repo.status}`}>
      <div className="repo-status">
        {repo.status === "indexing" && <span className="spinner small" />}
        {repo.status === "ready" && <span className="dot dot-ok" />}
        {repo.status === "error" && <span className="dot dot-bad" />}
        <span>
          {repo.status === "ready" &&
            `${repo.file_count ?? "?"} files · ${repo.chunk_count ?? "?"} chunks`}
          {repo.status === "ready" && repo.head_commit && (
            <code className="commit" title={repo.head_commit}>
              {" "}@ {repo.head_commit.slice(0, 8)}
            </code>
          )}
          {repo.status === "indexing" && (repo.message ?? "Indexing…")}
          {repo.status === "error" && "Indexing failed"}
        </span>
      </div>

      {repo.status !== "indexing" && repo.message && (
        <p className={`repo-message ${repo.status === "error" ? "error-text" : ""}`}>
          {repo.message}
        </p>
      )}

      {repo.status !== "indexing" && (
        <div className="repo-actions">
          {repo.indexed_at && (
            <span className="muted">
              Indexed {formatRelativeTime(repo.indexed_at)}
            </span>
          )}
          <span className="spacer" />
          {confirmDelete ? (
            <>
              <button className="text-button danger" onClick={onDelete}>
                Remove
              </button>
              <button
                className="text-button"
                onClick={() => setConfirmDelete(false)}
              >
                Keep
              </button>
            </>
          ) : (
            <>
              <button
                className="icon-button"
                onClick={() => onReindex(false)}
                title="Sync: download and embed only the files that changed"
                aria-label="Sync repository"
              >
                <Icon name="refresh" />
              </button>
              <button
                className="text-button"
                onClick={() => onReindex(true)}
                title="Rebuild the index from scratch"
              >
                Rebuild
              </button>
              <button
                className="icon-button"
                onClick={() => setConfirmDelete(true)}
                title="Remove repository and its chats"
                aria-label="Remove repository"
              >
                <Icon name="trash" />
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ChatItem({
  chat,
  active,
  busy,
  onSelect,
  onRename,
  onDelete,
}: {
  chat: ChatSummary;
  active: boolean;
  busy: boolean;
  onSelect: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(chat.title);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const save = () => {
    const value = title.trim();
    setEditing(false);

    if (value && value !== chat.title) onRename(value);
    else setTitle(chat.title);
  };

  if (editing) {
    return (
      <div className="chat-item active">
        <input
          className="chat-rename"
          autoFocus
          dir="auto"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          onBlur={save}
          onKeyDown={(event) => {
            if (event.key === "Enter") save();
            if (event.key === "Escape") {
              setTitle(chat.title);
              setEditing(false);
            }
          }}
        />
      </div>
    );
  }

  return (
    <div
      className={`chat-item ${active ? "active" : ""}`}
      onMouseLeave={() => setConfirmDelete(false)}
    >
      <button className="chat-item-main" onClick={onSelect} title={chat.title}>
        <span className="chat-title" dir="auto">
          {chat.title}
        </span>
        <span className="chat-meta">
          {busy ? "Answering…" : formatRelativeTime(chat.updated_at)}
        </span>
      </button>
      <div className="chat-item-actions">
        {confirmDelete ? (
          <button className="text-button danger" onClick={onDelete}>
            Delete
          </button>
        ) : (
          <>
            <button
              className="icon-button"
              onClick={() => {
                setTitle(chat.title);
                setEditing(true);
              }}
              aria-label="Rename chat"
              title="Rename"
            >
              <Icon name="edit" />
            </button>
            <button
              className="icon-button"
              onClick={() => setConfirmDelete(true)}
              aria-label="Delete chat"
              title="Delete"
              disabled={busy}
            >
              <Icon name="trash" />
            </button>
          </>
        )}
      </div>
    </div>
  );
}
