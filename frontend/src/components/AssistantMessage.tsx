import { useState } from "react";
import type { AnswerResult, Message, Source } from "../types";
import Icon from "./Icon";

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

  if (!result || !("sources" in result)) {
    return (
      <div className="message assistant-message">
        <p className="answer-text" dir="auto">
          {message.content}
        </p>
      </div>
    );
  }

  return <Answer message={message} result={result} />;
}

function Answer({ message, result }: { message: Message; result: AnswerResult }) {
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can be denied; nothing else to do
    }
  };

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

      {result.sources.length > 0 && (
        <div className="sources">
          <div className="sources-label">Sources</div>
          {result.sources.map((source, position) => (
            <SourceCard
              key={`${source.chunk_id ?? source.file_path}-${position}`}
              source={source}
              number={position + 1}
              highlighted={highlighted === position}
            />
          ))}
        </div>
      )}

      <div className="answer-footer">
        {result.related_identifiers.length > 0 && (
          <details className="identifiers">
            <summary>
              {result.related_identifiers.length} related identifier
              {result.related_identifiers.length === 1 ? "" : "s"}
            </summary>
            <div className="identifier-list">
              {result.related_identifiers.map((identifier) => (
                <code key={identifier}>{identifier}</code>
              ))}
            </div>
          </details>
        )}
        <span className="spacer" />
        <button className="ghost-button small" onClick={copy}>
          <Icon name={copied ? "check" : "copy"} />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
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
