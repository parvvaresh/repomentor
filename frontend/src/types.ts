export type RepoStatus = "indexing" | "ready" | "error";

export interface Repo {
  id: number;
  project_path: string;
  status: RepoStatus;
  message: string | null;
  file_count: number | null;
  chunk_count: number | null;
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

export interface Source {
  chunk_id: string | null;
  project_path: string;
  file_path: string;
  start_line: number;
  end_line: number;
  commit_id: string;
  citation: string;
  url: string;
  content: string;
}

export interface Fact {
  text: string;
  /** Position of the supporting source in `sources`. */
  source: number;
}

/** Result of the mentor agent. Older messages have no `agent` field. */
export interface AnswerResult {
  agent?: "mentor";
  related_identifiers: string[];
  facts: Fact[];
  sources: Source[];
  retrieval_only: boolean;
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
}

/** Result of an agent the frontend has no dedicated view for. */
export interface GenericResult {
  agent: string;
  answer?: string;
  sources?: Source[];
}

export type AgentResult = AnswerResult | CoderResult | GenericResult;

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
}

export interface ServerStatus {
  ollama_available: boolean;
  model: string;
  gitlab_configured: boolean;
  gitlab_base_url: string;
  default_top_k: number;
  max_top_k: number;
  agents: AgentInfo[];
  default_agent: string;
}
