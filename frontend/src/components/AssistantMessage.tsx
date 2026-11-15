import { useState } from "react";
import { api } from "../api";
import { formatRelativeTime } from "../time";
import type {
  AnswerResult,
  CoderResult,
  ExpertsResult,
  FileChange,
  GenericResult,
  Message,
  ReviewerResult,
  Source,
  TourResult,
} from "../types";
import Icon from "./Icon";
import Markdown from "./Markdown";

interface Props {
  message: Message;
  allowMergeRequests: boolean;
  onChange: (message: Message) => void;
  onError: (error: unknown) => void;
}

export default function AssistantMessage(props: Props) {
  const { message } = props;
  const result = message.result;

  if (result && "error" in result) {
    return (
      <div className="message assistant-message">
        <div className="answer-error">
          <Icon name="alert" />
          <div>
            <strong>Couldn't answer this question</strong>
            <p>{result.error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!result || !("sources" in result || "answer" in result)) {
    return (
      <div className="message assistant-message">
        <p className="answer-text" dir="auto">
          {message.content}
        </p>
      </div>
    );
  }

  // Pick the view for the agent that produced the answer. Messages saved
  // before agents existed have no `agent` and come from the mentor.
  const agent = "agent" in result ? result.agent : "mentor";

  switch (agent) {
    case "mentor":
      return <Answer message={message} result={result as AnswerResult} />;
    case "coder":
      return <CoderAnswer {...props} result={result as CoderResult} />;
    case "reviewer":
      return <ReviewerAnswer message={message} result={result as ReviewerResult} />;
    case "experts":
      return <ExpertsAnswer message={message} result={result as ExpertsResult} />;
    case "tour":
      return <TourAnswer message={message} result={result as TourResult} />;
    default:
      return <GenericAnswer message={message} result={result as GenericResult} />;
  }
}

// ---------------------------------------------------------------------------
// Mentor
// ---------------------------------------------------------------------------

const REFUSALS: Record<string, string> = {
  low_relevance:
    "Nothing in the repository looks related to this question, so I didn't try to answer it.",
  no_facts: "I couldn't find enough evidence in the retrieved repository sources.",
};

function Answer({ message, result }: { message: Message; result: AnswerResult }) {
  const [highlighted, setHighlighted] = useState<number | null>(null);

  return (
    <div className="message assistant-message">
      {result.retrieval_only ? (
        <p className="answer-intro muted">
          Retrieved {result.sources.length} source
          {result.sources.length === 1 ? "" : "s"} without calling the
          language model.
        </p>
      ) : result.facts.length > 0 ? (
        <ul className="facts">
          {result.facts.map((fact, position) => {
            const source = result.sources[fact.source];
            const start = fact.start_line ?? source?.start_line;
            const end = fact.end_line ?? source?.end_line;

            return (
              <li key={position}>
                <span dir="auto">{fact.text}</span>{" "}
                {source && (
                  <a
                    className="citation"
                    href={fact.url ?? source.url}
                    target="_blank"
                    rel="noreferrer"
                    onMouseEnter={() => setHighlighted(fact.source)}
                    onMouseLeave={() => setHighlighted(null)}
                    title={`${source.file_path} lines ${start}-${end}`}
                  >
                    {fact.source + 1}
                  </a>
                )}
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="answer-text muted">
          {REFUSALS[result.refusal ?? "no_facts"] ?? REFUSALS.no_facts}
        </p>
      )}

      <SourceList sources={result.sources} highlighted={highlighted} />

      <AnswerFooter
        text={message.content}
        identifiers={result.related_identifiers}
        note={
          result.dropped_facts
            ? `${result.dropped_facts} unsupported fact${result.dropped_facts === 1 ? "" : "s"} removed`
            : null
        }
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Coder
// ---------------------------------------------------------------------------

function CoderAnswer({
  message,
  result,
  allowMergeRequests,
  onChange,
  onError,
}: Props & { result: CoderResult }) {
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const changes = result.changes ?? [];

  return (
    <div className="message assistant-message">
      {result.retrieval_only ? (
        <p className="answer-intro muted">
          Retrieved {result.sources.length} source
          {result.sources.length === 1 ? "" : "s"} the coder would build on,
          without calling the language model.
        </p>
      ) : (
        <Markdown
          text={result.answer}
          sources={result.sources}
          onHighlight={setHighlighted}
        />
      )}

      {changes.length > 0 && (
        <ChangeList
          message={message}
          result={result}
          changes={changes}
          allowMergeRequests={allowMergeRequests}
          onChange={onChange}
          onError={onError}
        />
      )}

      <SourceList
        sources={result.sources}
        highlighted={highlighted}
        label={result.retrieval_only ? "Sources" : "Context"}
      />

      <AnswerFooter
        text={message.content}
        identifiers={result.related_identifiers}
        note={result.model ? `Written by ${result.model}` : null}
      />
    </div>
  );
}

function ChangeList({
  message,
  result,
  changes,
  allowMergeRequests,
  onChange,
  onError,
}: {
  message: Message;
  result: CoderResult;
  changes: FileChange[];
  allowMergeRequests: boolean;
  onChange: (message: Message) => void;
  onError: (error: unknown) => void;
}) {
  const [opening, setOpening] = useState(false);
  const applicable = changes.filter((change) => change.status !== "failed");
  const failedCheck = applicable.some((change) => change.check.ok === false);
  const created = result.merge_request_created;

  const open = async () => {
    if (
      failedCheck &&
      !window.confirm("A syntax check failed. Open the merge request anyway?")
    ) {
      return;
    }

    setOpening(true);

    try {
      const merge = await api.openMergeRequest(message.id, failedCheck);
      onChange({ ...message, result: { ...result, merge_request_created: merge } });
    } catch (error) {
      onError(error);
    } finally {
      setOpening(false);
    }
  };

  return (
    <div className="changes">
      <div className="sources-label">
        Changes · {applicable.length} file{applicable.length === 1 ? "" : "s"}
      </div>
      {changes.map((change) => (
        <ChangeCard key={change.path} change={change} />
      ))}
      {allowMergeRequests && applicable.length > 0 && (
        <div className="change-actions">
          {created ? (
            <a className="ghost-button" href={created.url} target="_blank" rel="noreferrer">
              <Icon name="external" />
              Merge request !{created.number}
            </a>
          ) : (
            <button className="primary-button" onClick={open} disabled={opening}>
              {opening ? "Opening…" : "Open merge request"}
            </button>
          )}
          <span className="muted small">
            Opens a new branch with these changes for review. Nothing is merged.
          </span>
        </div>
      )}
    </div>
  );
}

function ChangeCard({ change }: { change: FileChange }) {
  const [open, setOpen] = useState(change.status !== "failed");
  const check = change.check;

  return (
    <div className={`change change-${change.status}`}>
      <button className="change-header" onClick={() => setOpen((value) => !value)}>
        <span className={`badge badge-${change.status}`}>{change.status}</span>
        <span className="source-file">{change.path}</span>
        <span className="spacer" />
        {check.ok !== null && (
          <span
            className={`badge ${check.ok ? "badge-ok" : "badge-bad"}`}
            title={check.message}
          >
            {check.ok ? "syntax ok" : "syntax error"}
          </span>
        )}
      </button>
      {open && (
        <>
          {change.error && <p className="change-error error-text">{change.error}</p>}
          {check.ok === false && <p className="change-error error-text">{check.message}</p>}
          {change.diff && <DiffView diff={change.diff} />}
        </>
      )}
    </div>
  );
}

function DiffView({ diff }: { diff: string }) {
  return (
    <pre className="diff" dir="ltr">
      <code>
        {diff.split("\n").map((line, position) => {
          const kind = line.startsWith("+++") || line.startsWith("---")
            ? "meta"
            : line.startsWith("@@")
              ? "hunk"
              : line.startsWith("+")
                ? "add"
                : line.startsWith("-")
                  ? "del"
                  : "ctx";
          return (
            <span key={position} className={`diff-line diff-${kind}`}>
              {line || " "}
              {"\n"}
            </span>
          );
        })}
      </code>
    </pre>
  );
}

// ---------------------------------------------------------------------------
// Reviewer, experts, tour, others
// ---------------------------------------------------------------------------

function ReviewerAnswer({ message, result }: { message: Message; result: ReviewerResult }) {
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const request = result.merge_request;

  return (
    <div className="message assistant-message">
      <p className="answer-intro muted">
        {request ? (
          <>
            Review of{" "}
            <a href={request.url} target="_blank" rel="noreferrer">
              !{request.number} {request.title}
            </a>{" "}
            by {request.author}
          </>
        ) : (
          "Review of the pasted diff"
        )}
        {" · "}
        {result.files.length} file{result.files.length === 1 ? "" : "s"}
        {result.skipped_files.length > 0 &&
          ` (${result.skipped_files.length} more not reviewed)`}
      </p>
      <Markdown text={result.answer} sources={result.sources} onHighlight={setHighlighted} />
      <SourceList sources={result.sources} highlighted={highlighted} label="Conventions compared" />
      <AnswerFooter text={message.content} identifiers={[]} note={`Reviewed by ${result.model}`} />
    </div>
  );
}

function ExpertsAnswer({ message, result }: { message: Message; result: ExpertsResult }) {
  return (
    <div className="message assistant-message">
      {result.people.length === 0 ? (
        <p className="answer-text muted">
          No commit authors were found for the relevant files.
        </p>
      ) : (
        <ol className="people">
          {result.people.map((person) => (
            <li key={person.email || person.name} className="person">
              <div className="person-header">
                <strong>{person.name}</strong>
                {person.email && <span className="muted small">{person.email}</span>}
                <span className="spacer" />
                <span className="muted small">
                  {person.commits} commit{person.commits === 1 ? "" : "s"}
                  {person.last_commit && ` · last ${formatRelativeTime(person.last_commit)}`}
                </span>
              </div>
              <div className="person-files">
                {person.files.slice(0, 4).map((file) => (
                  <code key={file.path} title={`${file.commits} commits`}>
                    {file.path}
                  </code>
                ))}
              </div>
              <ul className="person-commits">
                {person.recent_commits.map((commit) => (
                  <li key={commit.sha}>
                    {commit.url ? (
                      <a href={commit.url} target="_blank" rel="noreferrer">
                        {commit.sha.slice(0, 8)}
                      </a>
                    ) : (
                      <code>{commit.sha.slice(0, 8)}</code>
                    )}{" "}
                    <span dir="auto">{commit.title}</span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      )}
      {result.failures.length > 0 && (
        <p className="muted small">
          History unavailable for {result.failures.length} file
          {result.failures.length === 1 ? "" : "s"}.
        </p>
      )}
      <SourceList sources={result.sources} highlighted={null} label="Files checked" />
      <AnswerFooter
        text={message.content}
        identifiers={[]}
        note="Ranked by number and recency of commits"
      />
    </div>
  );
}

function TourAnswer({ message, result }: { message: Message; result: TourResult }) {
  if (result.mode === "answer") {
    return <GenericAnswer message={message} result={result} />;
  }

  const directories = Object.entries(result.directories ?? {});

  return (
    <div className="message assistant-message">
      <Markdown text={result.overview ?? ""} sources={[]} onHighlight={() => {}} />

      {(result.learning_path ?? []).length > 0 && (
        <>
          <p className="md-heading">Learning path</p>
          <ol className="learning-path">
            {result.learning_path!.map((step) => (
              <li key={step.path}>
                <code>{step.path}</code> <span dir="auto">{step.why}</span>
              </li>
            ))}
          </ol>
        </>
      )}

      {directories.length > 0 && (
        <details className="architecture">
          <summary>Architecture map · {directories.length} directories</summary>
          <ul>
            {directories.map(([directory, node]) => (
              <li
                key={directory}
                style={{
                  paddingInlineStart: `${directory === "." ? 0 : (directory.split("/").length) * 14}px`,
                }}
              >
                <code>{directory === "." ? "(root)" : `${directory}/`}</code>{" "}
                <span className="muted">{node.summary}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      <AnswerFooter
        text={message.content}
        identifiers={[]}
        note={
          result.generated_at
            ? `Summaries from ${formatRelativeTime(result.generated_at)}`
            : null
        }
      />
    </div>
  );
}

/** Fallback for agents without a dedicated view: Markdown plus sources. */
function GenericAnswer({
  message,
  result,
}: {
  message: Message;
  result: GenericResult;
}) {
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const sources = result.sources ?? [];

  return (
    <div className="message assistant-message">
      <Markdown
        text={result.answer ?? message.content}
        sources={sources}
        onHighlight={setHighlighted}
      />
      <SourceList sources={sources} highlighted={highlighted} />
      <AnswerFooter text={message.content} identifiers={[]} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

export function SourceList({
  sources,
  highlighted,
  label = "Sources",
}: {
  sources: Source[];
  highlighted: number | null;
  label?: string;
}) {
  if (sources.length === 0) return null;

  return (
    <div className="sources">
      <div className="sources-label">{label}</div>
      {sources.map((source, position) => (
        <SourceCard
          key={`${source.chunk_id ?? source.file_path}-${position}`}
          source={source}
          number={position + 1}
          highlighted={highlighted === position}
        />
      ))}
    </div>
  );
}

function AnswerFooter({
  text,
  identifiers,
  note = null,
}: {
  text: string;
  identifiers: string[];
  note?: string | null;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can be denied; nothing else to do
    }
  };

  return (
    <div className="answer-footer">
      {identifiers.length > 0 && (
        <details className="identifiers">
          <summary>
            {identifiers.length} related identifier
            {identifiers.length === 1 ? "" : "s"}
          </summary>
          <div className="identifier-list">
            {identifiers.map((identifier) => (
              <code key={identifier}>{identifier}</code>
            ))}
          </div>
        </details>
      )}
      {note && <span className="answer-note muted">{note}</span>}
      <span className="spacer" />
      <button className="ghost-button small" onClick={copy}>
        <Icon name={copied ? "check" : "copy"} />
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

const KIND_LABELS: Record<string, string> = {
  commit: "commit",
  merge_request: "merge request",
  summary: "AI summary",
};

const WARNING_TEXT =
  "This source contains text that looks like instructions to an AI " +
  "(possible prompt injection). RepoMentor treats it as data, but check the " +
  "answer against the code yourself.";

function SourceCard({
  source,
  number,
  highlighted,
}: {
  source: Source;
  number: number;
  highlighted: boolean;
}) {
  const [open, setOpen] = useState(false);
  const lines = source.content.split("\n");
  const kind = source.kind ?? "code";
  const isCode = kind === "code";
  const host = source.provider === "github" ? "GitHub" : "GitLab";

  return (
    <div className={`source ${highlighted ? "highlighted" : ""}`}>
      <div className="source-header">
        <span className="source-number">{number}</span>
        <button
          className="source-path"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          title={open ? "Hide content" : "Show content"}
        >
          {!isCode && <span className="badge">{KIND_LABELS[kind] ?? kind}</span>}
          <span className="source-file">
            {isCode ? source.file_path : (source.title ?? source.file_path)}
          </span>
          {isCode && (
            <span className="source-lines">
              :{source.start_line}-{source.end_line}
            </span>
          )}
          {(source.warnings ?? []).length > 0 && (
            <span className="badge badge-bad" title={WARNING_TEXT}>
              <Icon name="alert" /> suspicious text
            </span>
          )}
        </button>
        <button
          className="icon-button"
          onClick={() => setOpen((value) => !value)}
          aria-label={open ? "Hide content" : "Show content"}
          title={open ? "Hide content" : "Show content"}
        >
          <Icon name="code" />
        </button>
        {source.url && (
          <a
            className="icon-button"
            href={source.url}
            target="_blank"
            rel="noreferrer"
            aria-label={`Open in ${host}`}
            title={
              isCode && source.commit_id
                ? `Open in ${host} at ${source.commit_id.slice(0, 8)}`
                : `Open in ${host}`
            }
          >
            <Icon name="external" />
          </a>
        )}
      </div>
      {open && (
        <pre className="source-code">
          <code>
            {lines.map((line, offset) => (
              <span className="code-line" key={offset}>
                {isCode && (
                  <span className="line-number">{source.start_line + offset}</span>
                )}
                {line || " "}
                {"\n"}
              </span>
            ))}
          </code>
        </pre>
      )}
    </div>
  );
}
