import type {
  Chat,
  ChatSummary,
  Message,
  QuestionOptions,
  Repo,
  ServerStatus,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch {
    throw new ApiError(
      "Cannot reach the RepoMentor server. Is `uvicorn server:app` running?",
      0,
    );
  }

  if (!response.ok) {
    let message = `Request failed (${response.status})`;

    try {
      const body = await response.json();

      if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail) && body.detail[0]?.msg) {
        message = body.detail[0].msg;
      }
    } catch {
      // Keep the generic message when the body is not JSON
    }

    throw new ApiError(message, response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export const api = {
  status: () => request<ServerStatus>("/status"),

  listRepos: () => request<Repo[]>("/repos"),
  addRepo: (projectPath: string) =>
    request<Repo>("/repos", {
      method: "POST",
      body: JSON.stringify({ project_path: projectPath }),
    }),
  reindexRepo: (repoId: number) =>
    request<Repo>(`/repos/${repoId}/reindex`, { method: "POST" }),
  deleteRepo: (repoId: number) =>
    request<void>(`/repos/${repoId}`, { method: "DELETE" }),

  listChats: (repoId: number) =>
    request<ChatSummary[]>(`/chats?repo_id=${repoId}`),
  createChat: (repoId: number) =>
    request<ChatSummary>("/chats", {
      method: "POST",
      body: JSON.stringify({ repo_id: repoId }),
    }),
  getChat: (chatId: number) => request<Chat>(`/chats/${chatId}`),
  renameChat: (chatId: number, title: string) =>
    request<{ id: number; title: string }>(`/chats/${chatId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteChat: (chatId: number) =>
    request<void>(`/chats/${chatId}`, { method: "DELETE" }),

  ask: (chatId: number, question: string, options: QuestionOptions) =>
    request<{ messages: Message[] }>(`/chats/${chatId}/messages`, {
      method: "POST",
      body: JSON.stringify({ question, ...options }),
    }),
};
