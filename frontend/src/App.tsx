import React, { useEffect, useRef, useState } from "react";
import {
  fetchPrograms,
  fetchSessionHistory,
  streamChat,
  submitFeedback,
} from "./api";
import { SafeMarkdown } from "./markdown";
import { ChatMessage, ProgramItem, RagStage } from "./types";

const SESSION_STORAGE_KEY = "session_id";
const MAX_QUESTION_CHARS = 1000;

const STAGE_LABELS: Record<RagStage, string> = {
  planning: "Анализ вопроса и истории…",
  retrieving: "Поиск по базе знаний AITU…",
  generating: "Формирование ответа…",
};

const EXAMPLE_QUESTIONS: Array<{ label: string; question: string; programCode?: string }> = [
  {
    label: "Сравнение SE и BDA",
    question: "Сравни дисциплины и карьерные перспективы программ Software Engineering и Big Data Analysis",
  },
  {
    label: "Навыки Cybersecurity",
    question: "Какие навыки и технологии изучают на программе Cybersecurity (6B06301)?",
    programCode: "6B06301",
  },
  {
    label: "Поступление и гранты",
    question: "Какие документы нужны для поступления в AITU и какие есть виды грантов?",
  },
  {
    label: "Академическая мобильность",
    question: "Как работает программа международной академической мобильности для студентов AITU?",
  },
];

function getStoredSessionId(): string | null {
  try {
    const raw = window.localStorage.getItem(SESSION_STORAGE_KEY);
    return raw ? raw.trim() || null : null;
  } catch {
    return null;
  }
}

function setStoredSessionId(sessionId: string | null): void {
  try {
    // Store ONLY session_id in localStorage per specification
    window.localStorage.clear();
    if (sessionId) {
      window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    }
  } catch {
    // Ignore storage quota / privacy mode errors
  }
}

export const App: React.FC = () => {
  const [programs, setPrograms] = useState<ProgramItem[]>([]);
  const [selectedProgram, setSelectedProgram] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(() => getStoredSessionId());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [currentStage, setCurrentStage] = useState<RagStage | null>(null);
  const [loadingHistory, setLoadingHistory] = useState<boolean>(false);
  const [bannerError, setBannerError] = useState<string | null>(null);

  // Feedback modal / inline comment state keyed by message id
  const [activeCommentMsgId, setActiveCommentMsgId] = useState<string | null>(null);
  const [commentDraft, setCommentDraft] = useState<string>("");
  const [pendingRating, setPendingRating] = useState<1 | -1>(1);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState<boolean>(false);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchPrograms()
      .then(setPrograms)
      .catch(() => {
        // Non-fatal if server starts up
      });
  }, []);

  useEffect(() => {
    const initialSession = getStoredSessionId();
    if (!initialSession) return;
    setLoadingHistory(true);
    fetchSessionHistory(initialSession)
      .then((data) => {
        if (!data) {
          setStoredSessionId(null);
          setSessionId(null);
          return;
        }
        setSessionId(data.session_id);
        setStoredSessionId(data.session_id);
        setMessages(
          data.messages.map((m) => ({
            id: m.id,
            localKey: m.id,
            role: m.role,
            content: m.content,
            program_code: m.program_code,
            finish_reason: m.finish_reason,
            created_at: m.created_at,
            feedback: m.feedback,
          }))
        );
      })
      .catch((err) => {
        setBannerError(err instanceof Error ? err.message : "Ошибка загрузки истории.");
      })
      .finally(() => setLoadingHistory(false));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStage]);

  const handleStop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  };

  const handleNewSession = () => {
    handleStop();
    setStoredSessionId(null);
    setSessionId(null);
    setMessages([]);
    setBannerError(null);
    setCurrentStage(null);
  };

  const sendQuestion = async (questionText: string, overrideProgram?: string) => {
    const trimmed = questionText.trim();
    if (!trimmed || isStreaming) return;
    if (trimmed.length > MAX_QUESTION_CHARS) {
      setBannerError(`Вопрос не должен превышать ${MAX_QUESTION_CHARS} символов.`);
      return;
    }

    setBannerError(null);
    const activeProgram =
      overrideProgram !== undefined ? overrideProgram : selectedProgram;
    if (overrideProgram !== undefined) {
      setSelectedProgram(overrideProgram);
    }

    const userKey = `user-${Date.now()}`;
    const assistantKey = `assistant-${Date.now() + 1}`;

    setMessages((prev) => [
      ...prev,
      {
        id: null,
        localKey: userKey,
        role: "user",
        content: trimmed,
        program_code: activeProgram || null,
      },
      {
        id: null,
        localKey: assistantKey,
        role: "assistant",
        content: "",
        program_code: activeProgram || null,
        streaming: true,
      },
    ]);
    setInput("");
    setIsStreaming(true);
    setCurrentStage("planning");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat({
        question: trimmed,
        sessionId,
        programCode: activeProgram || null,
        signal: controller.signal,
        onStatus: (stage) => {
          setCurrentStage(stage);
        },
        onToken: (delta) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.localKey === assistantKey
                ? { ...m, content: m.content + delta }
                : m
            )
          );
        },
        onDone: (donePayload) => {
          if (donePayload.session_id) {
            setSessionId(donePayload.session_id);
            setStoredSessionId(donePayload.session_id);
          }
          setMessages((prev) =>
            prev.map((m) =>
              m.localKey === assistantKey
                ? {
                    ...m,
                    id: donePayload.message_id,
                    finish_reason: donePayload.finish_reason,
                    streaming: false,
                  }
                : m
            )
          );
        },
        onError: (errPayload) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.localKey === assistantKey
                ? {
                    ...m,
                    streaming: false,
                    error: errPayload.message,
                  }
                : m
            )
          );
        },
      });
    } catch (err: any) {
      if (err?.name === "AbortError") {
        setMessages((prev) =>
          prev.map((m) =>
            m.localKey === assistantKey
              ? {
                  ...m,
                  streaming: false,
                  stopped: true,
                }
              : m
          )
        );
      } else {
        setMessages((prev) =>
          prev.map((m) =>
            m.localKey === assistantKey
              ? {
                  ...m,
                  streaming: false,
                  error:
                    err instanceof Error
                      ? err.message
                      : "Соединение прервано. Попробуйте ещё раз.",
                }
              : m
          )
        );
      }
    } finally {
      abortRef.current = null;
      setIsStreaming(false);
      setCurrentStage(null);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void sendQuestion(input);
  };

  const openFeedbackForm = (messageId: string, rating: 1 | -1, existingComment?: string | null) => {
    setActiveCommentMsgId(messageId);
    setPendingRating(rating);
    setCommentDraft(existingComment || "");
    setFeedbackError(null);
  };

  const handleFeedbackSave = async (messageId: string) => {
    setFeedbackSubmitting(true);
    setFeedbackError(null);
    try {
      await submitFeedback({
        messageId,
        rating: pendingRating,
        comment: commentDraft,
      });
      setMessages((prev) =>
        prev.map((m) =>
          m.id === messageId
            ? {
                ...m,
                feedback: {
                  rating: pendingRating,
                  comment: commentDraft.trim() || null,
                },
              }
            : m
        )
      );
      setActiveCommentMsgId(null);
      setCommentDraft("");
    } catch (err) {
      setFeedbackError(
        err instanceof Error ? err.message : "Не удалось отправить оценку."
      );
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-badge">AITU</div>
          <div>
            <h1 className="brand-title">AITUNATOR</h1>
            <p className="brand-subtitle">
              ИИ-консультант по образовательным программам и правилам Astana IT University
            </p>
          </div>
        </div>

        <div className="header-controls">
          <label className="program-select-wrap" htmlFor="program-select">
            <span className="sr-only">Образовательная программа</span>
            <select
              id="program-select"
              className="program-select"
              value={selectedProgram}
              onChange={(e) => setSelectedProgram(e.target.value)}
              disabled={isStreaming}
            >
              <option value="">Все программы AITU</option>
              {programs.map((p) => (
                <option key={p.program_code} value={p.program_code}>
                  {p.program_code} — {p.program_name}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className="btn-secondary"
            onClick={handleNewSession}
            title="Начать новую сессию"
          >
            Новый диалог
          </button>
        </div>
      </header>

      {bannerError && (
        <div className="banner-error" role="alert">
          <span>{bannerError}</span>
          <button
            type="button"
            className="banner-close"
            onClick={() => setBannerError(null)}
          >
            ✕
          </button>
        </div>
      )}

      <main className="chat-main">
        {loadingHistory ? (
          <div className="empty-state">Восстановление истории сессии…</div>
        ) : messages.length === 0 ? (
          <section className="welcome-card">
            <h2>Задайте вопрос о поступлении, обучении или программах AITU</h2>
            <p>
              Выберите образовательную программу сверху или нажмите на один из примеров ниже:
            </p>
            <div className="examples-grid">
              {EXAMPLE_QUESTIONS.map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="example-chip"
                  onClick={() =>
                    void sendQuestion(item.question, item.programCode ?? selectedProgram)
                  }
                >
                  <span className="example-chip-title">{item.label}</span>
                  <span className="example-chip-text">{item.question}</span>
                </button>
              ))}
            </div>
          </section>
        ) : (
          <div className="messages-list">
            {messages.map((msg) => (
              <article
                key={msg.localKey}
                className={`message-row message-${msg.role}`}
              >
                <div className="message-bubble">
                  <div className="message-meta">
                    <span className="message-author">
                      {msg.role === "user" ? "Вы" : "AITUNATOR"}
                    </span>
                    {msg.program_code && (
                      <span className="message-program-tag">{msg.program_code}</span>
                    )}
                    {msg.finish_reason === "guardrail" && (
                      <span className="message-guardrail-tag">Ограничение безопасности</span>
                    )}
                  </div>

                  {msg.role === "user" ? (
                    <div className="user-text">{msg.content}</div>
                  ) : (
                    <>
                      <SafeMarkdown content={msg.content} />
                      {msg.streaming && currentStage && (
                        <div className="stage-indicator">
                          <span className="pulse-dot" />
                          <span>{STAGE_LABELS[currentStage]}</span>
                        </div>
                      )}
                      {msg.stopped && (
                        <div className="stopped-note">Генерация остановлена пользователем.</div>
                      )}
                      {msg.error && (
                        <div className="message-error" role="alert">
                          {msg.error}
                        </div>
                      )}
                    </>
                  )}

                  {msg.role === "assistant" && !msg.streaming && msg.id && (
                    <div className="feedback-bar">
                      <span className="feedback-label">Оцените ответ:</span>
                      <button
                        type="button"
                        className={`feedback-btn ${
                          msg.feedback?.rating === 1 ? "active-up" : ""
                        }`}
                        onClick={() =>
                          openFeedbackForm(msg.id!, 1, msg.feedback?.comment)
                        }
                        aria-label="Полезный ответ"
                      >
                        👍 Полезно
                      </button>
                      <button
                        type="button"
                        className={`feedback-btn ${
                          msg.feedback?.rating === -1 ? "active-down" : ""
                        }`}
                        onClick={() =>
                          openFeedbackForm(msg.id!, -1, msg.feedback?.comment)
                        }
                        aria-label="Ответ требует доработки"
                      >
                        👎 Неточно
                      </button>

                      {activeCommentMsgId === msg.id && (
                        <div className="feedback-popover">
                          <input
                            type="text"
                            className="feedback-input"
                            placeholder="Комментарий (необязательно, без личных данных)…"
                            maxLength={500}
                            value={commentDraft}
                            onChange={(e) => setCommentDraft(e.target.value)}
                          />
                          <div className="feedback-actions">
                            <button
                              type="button"
                              className="btn-primary-sm"
                              disabled={feedbackSubmitting}
                              onClick={() => void handleFeedbackSave(msg.id!)}
                            >
                              {feedbackSubmitting ? "Сохранение…" : "Отправить"}
                            </button>
                            <button
                              type="button"
                              className="btn-ghost-sm"
                              onClick={() => setActiveCommentMsgId(null)}
                            >
                              Отмена
                            </button>
                          </div>
                          {feedbackError && (
                            <div className="feedback-error">{feedbackError}</div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </article>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </main>

      <footer className="chat-footer">
        <form className="composer-form" onSubmit={handleFormSubmit}>
          <div className="composer-input-wrap">
            <textarea
              className="composer-textarea"
              rows={2}
              maxLength={MAX_QUESTION_CHARS}
              placeholder="Напишите вопрос об образовательных программах или поступлении в AITU…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void sendQuestion(input);
                }
              }}
              disabled={isStreaming}
            />
            <div className="char-counter">
              {input.length} / {MAX_QUESTION_CHARS}
            </div>
          </div>

          <div className="composer-buttons">
            {isStreaming ? (
              <button
                type="button"
                className="btn-stop"
                onClick={handleStop}
              >
                ■ Стоп
              </button>
            ) : (
              <button
                type="submit"
                className="btn-send"
                disabled={!input.trim() || input.trim().length > MAX_QUESTION_CHARS}
              >
                Отправить
              </button>
            )}
          </div>
        </form>
      </footer>
    </div>
  );
};
