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

export interface AnswerResult {
  related_identifiers: string[];
  facts: Fact[];
  sources: Source[];
  retrieval_only: boolean;
}

export interface QuestionOptions {
  top_k: number;
  retrieval_only: boolean;
}

export interface ErrorResult {
  error: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  result: AnswerResult | ErrorResult | QuestionOptions | null;
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

export interface ServerStatus {
  ollama_available: boolean;
  model: string;
  gitlab_configured: boolean;
  gitlab_base_url: string;
  default_top_k: number;
  max_top_k: number;
}
