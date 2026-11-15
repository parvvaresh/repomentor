import { useState, type FormEvent } from "react";
import type { Credentials } from "../types";

const LABELS = { gitlab: "GitLab", github: "GitHub" } as const;

export default function SignIn({
  providers,
  initialError,
  onSignIn,
}: {
  providers: ("gitlab" | "github")[];
  initialError: string | null;
  onSignIn: (credentials: Credentials) => Promise<void>;
}) {
  const [provider, setProvider] = useState<"gitlab" | "github">(
    providers[0] ?? "gitlab",
  );
  const [token, setToken] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(initialError);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!token.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      await onSignIn({ provider, token: token.trim() });
    } catch (value) {
      setError(value instanceof Error ? value.message : String(value));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="sign-in">
      <form className="sign-in-card" onSubmit={submit}>
        <div className="brand">
          <span className="brand-mark">R</span>
          <span>RepoMentor</span>
        </div>
        <h1>Sign in</h1>
        <p className="muted">
          Use a personal access token with read access. RepoMentor shows only
          the repositories your token can read, and keeps your chats private.
          The token stays in this browser and is never stored on the server.
        </p>

        {providers.length > 1 && (
          <div className="segmented" role="radiogroup" aria-label="Host">
            {providers.map((value) => (
              <button
                type="button"
                key={value}
                role="radio"
                aria-checked={provider === value}
                className={provider === value ? "is-active" : ""}
                onClick={() => setProvider(value)}
              >
                {LABELS[value]}
              </button>
            ))}
          </div>
        )}

        <input
          type="password"
          autoFocus
          autoComplete="off"
          value={token}
          onChange={(event) => setToken(event.target.value)}
          placeholder={`${LABELS[provider]} personal access token`}
          aria-label="Access token"
          disabled={submitting}
        />
        <button className="primary-button" disabled={submitting || !token.trim()}>
          {submitting ? "Checking…" : "Sign in"}
        </button>
        {error && <p className="form-hint error-text">{error}</p>}
      </form>
    </div>
  );
}
