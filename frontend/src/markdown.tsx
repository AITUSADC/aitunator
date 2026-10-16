import React from "react";
import ReactMarkdown from "react-markdown";

/**
 * Safe Markdown renderer using react-markdown (per TZ Section 6):
 * - Raw HTML is disabled by default in react-markdown (no rehype-raw plugin).
 * - External/source links and bracketed [n] citations are stripped/rendered as plain text
 *   per the project owner decision in aitunator_status_and_plan.pdf & docs/contract.md.
 */
function sanitizeContent(text: string): string {
  // Replace markdown links [label](url) with plain label
  let cleaned = text.replace(/\[([^\]]+)\]\((?:https?:\/\/|\/)[^)]+\)/g, "$1");
  // Remove standalone URLs so source links are never displayed
  cleaned = cleaned.replace(/https?:\/\/[^\s)>\]]+/g, "");
  return cleaned;
}

export const SafeMarkdown: React.FC<{ content: string }> = ({ content }) => {
  if (!content) return null;
  const sanitized = sanitizeContent(content);

  return (
    <div className="md-content">
      <ReactMarkdown
        skipHtml={true}
        components={{
          a: ({ children }) => <span>{children}</span>,
          p: ({ children }) => <p className="md-paragraph">{children}</p>,
          ul: ({ children }) => <ul className="md-list">{children}</ul>,
          ol: ({ children }) => <ol className="md-list">{children}</ol>,
          code: ({ className, children }) => {
            const isBlock = Boolean(className);
            return isBlock ? (
              <pre className="md-code-block">
                <code>{children}</code>
              </pre>
            ) : (
              <code className="md-inline-code">{children}</code>
            );
          },
          table: ({ children }) => (
            <div className="md-table-wrap">
              <table className="md-table">{children}</table>
            </div>
          ),
        }}
      >
        {sanitized}
      </ReactMarkdown>
    </div>
  );
};
