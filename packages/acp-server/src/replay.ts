import type { SessionNotification } from '@agentclientprotocol/sdk';
import type { ContentPart, ContextMessage, ToolCall } from '@moonshot-ai/agent-core-v2';
import type { ToolResultEvent } from '@moonshot-ai/agent-core-v2/events';
import type { ToolCallStartedEvent } from '@moonshot-ai/agent-core-v2/agent/toolExecutor/toolExecutorEvents';

import {
  assistantDeltaToSessionUpdate,
  thinkingDeltaToSessionUpdate,
  toolCallStartToSessionUpdate,
  completedToolToSessionUpdate,
  contextMessageToSessionUpdates,
  legacyMessageId,
  planFromDisplayBlock,
} from './events-map';

/**
 * Project a persisted context history into an ordered batch of ACP
 * `session/update` notifications. Pure — no IO, never throws on a single
 * malformed message (it is skipped).
 */
export interface AcpHistory {
  readonly messages: readonly ContextMessage[];
  readonly toolCalls?: ReadonlyMap<string, ToolCallStartedEvent>;
  readonly toolResults?: ReadonlyMap<string, ToolResultEvent>;
}

export function projectHistoryToSessionUpdates(
  sessionId: string,
  messages: readonly ContextMessage[],
  toolCalls?: AcpHistory['toolCalls'],
  toolResults?: AcpHistory['toolResults'],
): SessionNotification[] {
  const out: SessionNotification[] = [];
  let turnId = 0;
  const toolCallTurnIds = new Map<string, number>();

  for (const [position, message] of messages.entries()) {
    switch (message.role) {
      case 'user':
      case 'system':
        out.push(...contextMessageToSessionUpdates(sessionId, message, legacyMessageId(message, position)));
        break;
      case 'assistant': {
        turnId += 1;
        for (const part of message.content) {
          const update = assistantContentPartToUpdate(part, sessionId, turnId, message.id ?? legacyMessageId(message, position));
          if (update !== null) out.push(update);
        }
        for (const toolCall of message.toolCalls ?? []) {
          const recorded = toolCalls?.get(toolCall.id);
          const callTurnId = recorded?.turnId ?? turnId;
          toolCallTurnIds.set(toolCall.id, callTurnId);
          out.push(recorded === undefined
            ? syntheticToolCall(sessionId, callTurnId, toolCall)
            : toolCallStartToSessionUpdate(sessionId, recorded));
          if (recorded?.display !== undefined) {
            const plan = planFromDisplayBlock(sessionId, callTurnId, recorded.display);
            if (plan !== null) out.push(plan);
          }
        }
        break;
      }
      case 'tool': {
        const update = toolMessageToUpdate(message, sessionId, toolCallTurnIds, toolCalls?.get(message.toolCallId ?? ''), toolResults?.get(message.toolCallId ?? ''));
        if (update !== null) out.push(update);
        break;
      }
    }
  }
  return out;
}

function assistantContentPartToUpdate(
  part: ContentPart,
  sessionId: string,
  turnId: number,
  messageId?: string,
): SessionNotification | null {
  if (part.type === 'text' && part.text) {
    return assistantDeltaToSessionUpdate(sessionId, {
      type: 'assistant.delta',
      turnId,
      delta: part.text,
    }, messageId);
  }
  if (part.type === 'think' && part.think && part.hidden !== true) {
    return thinkingDeltaToSessionUpdate(sessionId, {
      type: 'thinking.delta',
      turnId,
      delta: part.think,
    }, messageId);
  }
  // image_url / audio_url / video_url belong to the user-input side and ACP
  // has no dedicated assistant-media chunk; skip them.
  return null;
}

function syntheticToolCall(
  sessionId: string,
  turnId: number,
  toolCall: ToolCall,
): SessionNotification {
  return toolCallStartToSessionUpdate(sessionId, {
    type: 'tool.call.started',
    turnId,
    toolCallId: toolCall.id,
    name: toolCall.name,
    args: parseToolCallArguments(toolCall.arguments),
  });
}

function toolMessageToUpdate(
  message: ContextMessage,
  sessionId: string,
  toolCallTurnIds: ReadonlyMap<string, number>,
  recorded?: ToolCallStartedEvent,
  result?: ToolResultEvent,
): SessionNotification | null {
  const rawToolCallId = message.toolCallId;
  if (!rawToolCallId) {
    // Tool result with no correlation id — skip rather than crash; the
    // on-disk session is the source of truth and we cannot synthesize a
    // missing id.
    return null;
  }
  const turnId = toolCallTurnIds.get(rawToolCallId);
  if (turnId === undefined) {
    // The matching assistant message was not in this history slice (e.g. the
    // session was compacted). Skip — emitting an update for a tool_call the
    // client never saw would orphan the card.
    return null;
  }
  return completedToolToSessionUpdate(sessionId, result ?? {
    type: 'tool.result',
    turnId,
    toolCallId: rawToolCallId,
    isError: message.isError,
    output: message.content,
  }, recorded);
}

/**
 * Parse a tool call's `arguments` field (a JSON string or `null`) into the
 * structured object expected by {@link toolCallStartToSessionUpdate}. Falls
 * back to the raw string when the payload is not valid JSON.
 */
function parseToolCallArguments(rawArguments: string | null): unknown {
  if (rawArguments === null || rawArguments === '') return {};
  try {
    return JSON.parse(rawArguments);
  } catch {
    return rawArguments;
  }
}
