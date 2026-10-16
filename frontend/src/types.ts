export type RagStage = "planning" | "retrieving" | "generating";

export interface ProgramItem {
  program_code: string;
  program_slug: string;
  program_name: string;
  school: string;
}

export interface MessageFeedback {
  rating: 1 | -1;
  comment: string | null;
}

export interface ChatMessage {
  id: string | null;
  localKey: string;
  role: "user" | "assistant";
  content: string;
  program_code?: string | null;
  finish_reason?: string | null;
  created_at?: string;
  streaming?: boolean;
  stopped?: boolean;
  error?: string | null;
  feedback?: MessageFeedback | null;
}

export interface SessionHistoryResponse {
  session_id: string;
  created_at: string;
  last_active_at: string;
  messages: Array<{
    id: string;
    role: "user" | "assistant";
    content: string;
    program_code: string | null;
    finish_reason: string | null;
    created_at: string;
    feedback: MessageFeedback | null;
  }>;
}
