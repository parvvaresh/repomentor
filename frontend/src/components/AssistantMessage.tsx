import { useState } from "react";
import type {
  AnswerResult,
  CoderResult,
  GenericResult,
  Message,
  Source,
} from "../types";
import Icon from "./Icon";
import Markdown from "./Markdown";

export default function AssistantMessage({ message }: { message: Message }) {
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

  if (agent === "mentor") {
    return <Answer message={message} result={result as AnswerResult} />;
  }

  if (agent === "coder") {
    return <CoderAnswer message={message} result={result as CoderResult} />;
  }

  return <GenericAnswer message={message} result={result as GenericResult} />;
}

function CoderAnswer({
  message,
  result,
}: {
  message: Message;
  result: CoderResult;
}) {
  const [highlighted, setHighlighted] = useState<number | null>(null);

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

function SourceList({
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

            return (
              <li key={position}>
                <span dir="auto">{fact.text}</span>{" "}
                {source && (
                  <a
                    className="citation"
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    onMouseEnter={() => setHighlighted(fact.source)}
                    onMouseLeave={() => setHighlighted(null)}
                    title={`${source.file_path} lines ${source.start_line}-${source.end_line}`}
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
          I couldn't find enough evidence in the retrieved repository sources.
        </p>
      )}

      <SourceList sources={result.sources} highlighted={highlighted} />

      <AnswerFooter
        text={message.content}
        identifiers={result.related_identifiers}
      />
    </div>
  );
}

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

  return (
    <div className={`source ${highlighted ? "highlighted" : ""}`}>
      <div className="source-header">
        <span className="source-number">{number}</span>
        <button
          className="source-path"
          onClick={() => setOpen((value) => !value)}
          aria-expanded={open}
          title={open ? "Hide code" : "Show code"}
        >
          <span className="source-file">{source.file_path}</span>
          <span className="source-lines">
            :{source.start_line}-{source.end_line}
          </span>
        </button>
        <button
          className="icon-button"
          onClick={() => setOpen((value) => !value)}
          aria-label={open ? "Hide code" : "Show code"}
          title={open ? "Hide code" : "Show code"}
        >
          <Icon name="code" />
        </button>
        <a
          className="icon-button"
          href={source.url}
          target="_blank"
          rel="noreferrer"
          aria-label="Open in GitLab"
          title={`Open in GitLab at ${source.commit_id.slice(0, 8)}`}
        >
          <Icon name="external" />
        </a>
      </div>
      {open && (
        <pre className="source-code">
          <code>
            {lines.map((line, offset) => (
              <span className="code-line" key={offset}>
                <span className="line-number">{source.start_line + offset}</span>
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
