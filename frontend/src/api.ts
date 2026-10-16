import { ProgramItem, RagStage, SessionHistoryResponse } from "./types";

export async function fetchPrograms(): Promise<ProgramItem[]> {
  const resp = await fetch("/api/programs");
  if (!resp.ok) {
    throw new Error("Не удалось загрузить список образовательных программ.");
  }
  const data = await resp.json();
  return Array.isArray(data.programs) ? data.programs : [];
}

export async function fetchSessionHistory(
  sessionId: string
): Promise<SessionHistoryResponse | null> {
  const resp = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (resp.status === 404) {
    return null;
  }
  if (!resp.ok) {
    throw new Error("Не удалось восстановить историю диалога.");
  }
  return resp.json();
}

export async function submitFeedback(params: {
  messageId: string;
  rating: 1 | -1;
  comment?: string;
}): Promise<void> {
  const resp = await fetch("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message_id: params.messageId,
      rating: params.rating,
      comment: params.comment?.trim() || null,
    }),
  });
  if (!resp.ok) {
    const errBody = await resp.json().catch(() => ({}));
    throw new Error(
      errBody.message || errBody.detail || "Не удалось сохранить оценку ответа."
    );
  }
}

export interface StreamChatOptions {
  question: string;
  sessionId: string | null;
  programCode: string | null;
  signal: AbortSignal;
  onStatus: (stage: RagStage) => void;
  onToken: (text: string) => void;
  onDone: (payload: {
    session_id: string;
    message_id: string | null;
    finish_reason: string;
  }) => void;
  onError: (payload: {
    code: string;
    message: string;
    retryable?: boolean;
  }) => void;
}

export async function streamChat(options: StreamChatOptions): Promise<void> {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: options.question,
      session_id: options.sessionId,
      program_code: options.programCode,
    }),
    signal: options.signal,
  });

  if (!resp.ok) {
    const errBody = await resp.json().catch(() => ({}));
    options.onError({
      code: errBody.code || `http_${resp.status}`,
      message:
        errBody.message ||
        errBody.detail ||
        "Ошибка запроса к серверу. Попробуйте ещё раз.",
      retryable: Boolean(errBody.retryable ?? true),
    });
    return;
  }

  if (!resp.body) {
    options.onError({
      code: "empty_stream",
      message: "Сервер не вернул поток данных.",
      retryable: true,
    });
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const processBlock = (block: string) => {
    const trimmed = block.trim();
    if (!trimmed) return;
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of trimmed.split(/\r?\n/)) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trim());
      }
    }
    if (dataLines.length === 0) return;
    const rawJson = dataLines.join("\n");
    let parsed: any;
    try {
      parsed = JSON.parse(rawJson);
    } catch {
      return;
    }

    if (eventName === "status" && parsed.stage) {
      options.onStatus(parsed.stage as RagStage);
    } else if (eventName === "token" && typeof parsed.text === "string") {
      options.onToken(parsed.text);
    } else if (eventName === "done") {
      options.onDone({
        session_id: String(parsed.session_id || ""),
        message_id: parsed.message_id ? String(parsed.message_id) : null,
        finish_reason: String(parsed.finish_reason || "stop"),
      });
    } else if (eventName === "error") {
      options.onError({
        code: String(parsed.code || "stream_error"),
        message: String(parsed.message || "Ошибка при генерации ответа."),
        retryable: Boolean(parsed.retryable ?? true),
      });
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      processBlock(part);
    }
  }
  if (buffer.trim()) {
    processBlock(buffer);
  }
}
