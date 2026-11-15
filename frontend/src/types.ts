export type RepoStatus = "indexing" | "ready" | "error";
export type Provider = "gitlab" | "github" | "local";

export interface SyncReport {
  downloaded: number;
  reused: number;
  failed: number;
  reused_vectors: number;
  incremental: boolean;
  changed: number;
  removed: number;
}

export interface Repo {
  id: number;
  provider: Provider;
  project_path: string;
  status: RepoStatus;
  message: string | null;
  file_count: number | null;
  chunk_count: number | null;
  head_commit: string | null;
  last_sync: SyncReport | null;
  indexed_at: string | null;
  created_at: string;
}

export interface ChatSummary {
  id: number;
  repo_id: number;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export type SourceKind = "code" | "commit" | "merge_request" | "summary";

export interface Source {
  chunk_id: string | null;
  kind?: SourceKind;
  provider?: Provider;
  project_path: string;
  file_path: string;
  title?: string | null;
  start_line: number;
  end_line: number;
  commit_id: string;
  symbols?: string[];
  citation: string;
  url: string;
  content: string;
  /** Prompt-injection patterns found in the source text. */
  warnings?: string[];
}

export interface Fact {
  text: string;
  /** Position of the supporting source in `sources`. */
  source: number;
  /** Narrowed line range and link; older answers only have `source`. */
  start_line?: number;
  end_line?: number;
  citation?: string;
  url?: string;
}

/** Result of the mentor agent. Older messages have no `agent` field. */
export interface AnswerResult {
  agent?: "mentor";
  related_identifiers: string[];
  facts: Fact[];
  sources: Source[];
  retrieval_only: boolean;
  retrieval_method?: string;
  refusal?: "low_relevance" | "no_facts" | null;
  dropped_facts?: number;
}

export interface SyntaxCheck {
  ok: boolean | null;
  message: string;
}

export interface FileChange {
  path: string;
  status: "modified" | "created" | "failed";
  diff: string;
  content: string;
  check: SyntaxCheck;
  error: string | null;
}

export interface CreatedMergeRequest {
  url: string;
  number: number;
}

/** Result of the coder agent: Markdown that cites sources as [n]. */
export interface CoderResult {
  agent: "coder";
  answer: string;
  sources: Source[];
  /** Positions in `sources` the answer refers to. */
  cited_sources: number[];
  related_identifiers: string[];
  retrieval_only: boolean;
  model: string | null;
  changes?: FileChange[];
  base_commit?: string | null;
  merge_request_created?: CreatedMergeRequest;
}

export interface ReviewerResult {
  agent: "reviewer";
  answer: string;
  sources: Source[];
  cited_sources: number[];
  merge_request: {
    number: number;
    title: string;
    url: string;
    author: string;
  } | null;
  files: string[];
  skipped_files: string[];
  model: string;
}

export interface Person {
  name: string;
  email: string;
  score: number;
  commits: number;
  last_commit: string;
  files: { path: string; commits: number }[];
  recent_commits: {
    sha: string;
    title: string;
    url: string;
    date: string;
    file: string;
  }[];
}

export interface ExpertsResult {
  agent: "experts";
  people: Person[];
  sources: Source[];
  related_identifiers: string[];
  files_checked: number;
  failures: string[];
  answer: string;
}

export interface TourResult {
  agent: "tour";
  mode: "tour" | "answer";
  answer: string;
  sources: Source[];
  cited_sources: number[];
  overview?: string;
  learning_path?: { path: string; why: string }[];
  directories?: Record<
    string,
    { summary: string; files: string[]; children: string[] }
  >;
  generated_at?: string;
}

/** Result of an agent the frontend has no dedicated view for. */
export interface GenericResult {
  agent: string;
  answer?: string;
  sources?: Source[];
}

export type AgentResult =
  | AnswerResult
  | CoderResult
  | ReviewerResult
  | ExpertsResult
  | TourResult
  | GenericResult;

export interface QuestionOptions {
  top_k: number;
  retrieval_only: boolean;
  agent: string;
}

export interface ErrorResult {
  error: string;
  agent?: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  result: AgentResult | ErrorResult | QuestionOptions | null;
  created_at: string;
}

export interface Chat {
  id: number;
  repo_id: number;
  title: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
}

export interface AgentInfo {
  id: string;
  name: string;
  description: string;
  examples: string[];
  working_label: string;
  placeholder: string | null;
}

export interface ServerStatus {
  ollama_available: boolean;
  model: string;
  providers: Provider[];
  gitlab_configured: boolean;
  gitlab_base_url: string;
  github_web_url: string;
  auth_mode: "none" | "token";
  allow_merge_requests: boolean;
  default_top_k: number;
  max_top_k: number;
  agents: AgentInfo[];
  default_agent: string;
}

export interface Credentials {
  provider: "gitlab" | "github";
  token: string;
}

export interface CurrentUser {
  id: string;
  provider: Provider | null;
  username: string;
  name: string;
}

/** Events streamed while an agent answers. */
export type StreamEvent =
  | { type: "user"; message: Message }
  | { type: "status"; label: string }
  | { type: "sources"; sources: Source[] }
  | { type: "delta"; text: string }
  | { type: "fact"; fact: Fact; source: Source }
  | { type: "message"; message: Message };

/** Short display name: local checkouts are absolute paths, so show the folder. */
export function repoName(repo: Repo): string {
  if (repo.provider !== "local") return repo.project_path;
  const parts = repo.project_path.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? repo.project_path;
}
