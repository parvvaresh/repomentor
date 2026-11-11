import { useState, type ReactNode } from "react";
import type { Source } from "../types";
import Icon from "./Icon";

/**
 * A small, dependency-free renderer for the Markdown that agents write:
 * fenced code blocks, headings, bullet and numbered lists, paragraphs,
 * **bold**, `inline code` and [n] references to retrieved sources.
 * Everything is rendered as React text, so model output cannot inject HTML.
 */

interface Props {
  text: string;
  sources: Source[];
  onHighlight: (source: number | null) => void;
}

type Block =
  | { kind: "code"; language: string; code: string }
  | { kind: "heading"; level: number; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "paragraph"; text: string };

const FENCE = /^\s*```\s*([\w+#.-]*)\s*$/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^\s*[-*]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\[\d+\])/g;

function parseBlocks(text: string): Block[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ kind: "paragraph", text: paragraph.join("\n") });
      paragraph = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(FENCE);

    if (fence) {
      flushParagraph();
      const code: string[] = [];
      i++;
      // An unclosed fence (truncated output) runs to the end
      while (i < lines.length && !FENCE.test(lines[i])) {
        code.push(lines[i]);
        i++;
      }
      blocks.push({ kind: "code", language: fence[1], code: code.join("\n") });
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      flushParagraph();
      blocks.push({
        kind: "heading",
        level: heading[1].length,
        text: heading[2],
      });
      continue;
    }

    const bullet = line.match(BULLET);
    const numbered = line.match(NUMBERED);
    const item = bullet ?? numbered;

    if (item) {
      flushParagraph();
      const ordered = numbered !== null && bullet === null;
      const last = blocks[blocks.length - 1];

      if (last?.kind === "list" && last.ordered === ordered) {
        last.items.push(item[1]);
      } else {
        blocks.push({ kind: "list", ordered, items: [item[1]] });
      }
      continue;
    }

    if (line.trim() === "") {
      flushParagraph();
    } else {
      paragraph.push(line);
    }
  }

  flushParagraph();
  return blocks;
}

function Inline({ text, sources, onHighlight }: Props) {
  const parts = text.split(INLINE);

  return (
    <>
      {parts.map((part, position): ReactNode => {
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
          return <code key={position}>{part.slice(1, -1)}</code>;
        }

        if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
          return <strong key={position}>{part.slice(2, -2)}</strong>;
        }

        const reference = part.match(/^\[(\d+)\]$/);
        if (reference) {
          const index = Number(reference[1]) - 1;
          const source = sources[index];

          if (source) {
            return (
              <a
                key={position}
                className="citation"
                href={source.url}
                target="_blank"
                rel="noreferrer"
                onMouseEnter={() => onHighlight(index)}
                onMouseLeave={() => onHighlight(null)}
                title={`${source.file_path} lines ${source.start_line}-${source.end_line}`}
              >
                {index + 1}
              </a>
            );
          }
        }

        return part;
      })}
    </>
  );
}

function CodeBlock({ language, code }: { language: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard access can be denied; nothing else to do
    }
  };

  return (
    <div className="md-code">
      <div className="md-code-header">
        <span>{language || "code"}</span>
        <button className="ghost-button small" onClick={copy}>
          <Icon name={copied ? "check" : "copy"} />
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre dir="ltr">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export default function Markdown(props: Props) {
  const blocks = parseBlocks(props.text);
  const inline = (text: string) => (
    <Inline text={text} sources={props.sources} onHighlight={props.onHighlight} />
  );

  return (
    <div className="markdown">
      {blocks.map((block, position) => {
        switch (block.kind) {
          case "code":
            return (
              <CodeBlock
                key={position}
                language={block.language}
                code={block.code}
              />
            );
          case "heading":
            return (
              <p key={position} className="md-heading" dir="auto">
                {inline(block.text)}
              </p>
            );
          case "list": {
            const List = block.ordered ? "ol" : "ul";
            return (
              <List key={position}>
                {block.items.map((item, index) => (
                  <li key={index} dir="auto">
                    {inline(item)}
                  </li>
                ))}
              </List>
            );
          }
          default:
            return (
              <p key={position} dir="auto">
                {inline(block.text)}
              </p>
            );
        }
      })}
    </div>
  );
}
