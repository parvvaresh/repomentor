import type {
  Chat,
  ChatSummary,
  CreatedMergeRequest,
  Credentials,
  CurrentUser,
  Message,
  Provider,
  QuestionOptions,
  Repo,
  ServerStatus,
  StreamEvent,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

let credentials: Credentials | null = null;

/** Sign-in token sent with every request when the server uses AUTH_MODE=token. */
export function setCredentials(value: Credentials | null) {
  credentials = value;
}

function headers(extra: HeadersInit = {}): HeadersInit {
  const result: Record<string, string> = {
    "Content-Type": "application/json",
    ...(extra as Record<string, string>),
  };

  if (credentials) {
    result.Authorization = `Bearer ${credentials.token}`;
    result["X-RepoMentor-Provider"] = credentials.provider;
  }

  return result;
}

const UNREACHABLE =
  "Cannot reach the RepoMentor server. Is `python -m repomentor serve` running?";

async function errorFrom(response: Response): Promise<ApiError> {
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

  return new ApiError(message, response.status);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`/api${path}`, { ...init, headers: headers(init.headers) });
  } catch {
    throw new ApiError(UNREACHABLE, 0);
  }

  if (!response.ok) {
    throw await errorFrom(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

/**
 * POST a question and read the answer as Server-Sent Events. EventSource
 * cannot send a body or headers, so the stream is parsed by hand.
 */
async function askStream(
  chatId: number,
  question: string,
  options: QuestionOptions,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  let response: Response;

  try {
    response = await fetch(`/api/chats/${chatId}/messages/stream`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ question, ...options }),
    });
  } catch {
    throw new ApiError(UNREACHABLE, 0);
  }

  if (!response.ok || !response.body) {
    throw await errorFrom(response);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatch = (block: string) => {
    let kind = "message";
    const data: string[] = [];

    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) kind = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }

    if (data.length === 0) return;
    const payload = JSON.parse(data.join("\n"));

    if (kind === "user" || kind === "message") {
      onEvent({ type: kind, message: payload as Message });
    } else {
      onEvent(payload as StreamEvent);
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      dispatch(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");
    }
  }

  if (buffer.trim()) dispatch(buffer);
}

export const api = {
  status: () => request<ServerStatus>("/status"),
  me: () => request<CurrentUser>("/me"),

  listRepos: () => request<Repo[]>("/repos"),
  addRepo: (projectPath: string, provider: Provider | null) =>
    request<Repo>("/repos", {
      method: "POST",
      body: JSON.stringify({ project_path: projectPath, provider }),
    }),
  reindexRepo: (repoId: number, full = false) =>
    request<Repo>(`/repos/${repoId}/reindex${full ? "?full=true" : ""}`, {
      method: "POST",
    }),
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

  askStream,

  openMergeRequest: (messageId: number, force = false) =>
    request<CreatedMergeRequest>(`/messages/${messageId}/merge-request`, {
      method: "POST",
      body: JSON.stringify({ force }),
    }),
};
