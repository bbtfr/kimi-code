import { createHash } from 'node:crypto';
import { RequestError } from '@agentclientprotocol/sdk';
import type { ContextMessage } from '@moonshot-ai/agent-core-v2';
import { isAbsolute } from 'node:path';

import type {
  AvailableCommand,
  PlanEntry,
  PlanEntryStatus,
  SessionConfigOption,
  SessionNotification,
  ToolCallContent,
  ToolCallLocation,
  ToolKind,
} from '@agentclientprotocol/sdk';
import type { ToolResultEvent } from '@moonshot-ai/agent-core-v2/events';
import type {
  AssistantDeltaEvent,
  ThinkingDeltaEvent,
  TurnEndReason,
} from '@moonshot-ai/agent-core-v2/agent/loop/turnEvents';
import type {
  ToolCallDeltaEvent,
  ToolCallStartedEvent,
  ToolProgressEvent,
} from '@moonshot-ai/agent-core-v2/agent/toolExecutor/toolExecutorEvents';
import type { ToolInputDisplay } from '@moonshot-ai/agent-core-v2/tool/toolInputDisplay';

import { displayBlockToAcpContent, toolResultToAcpContent } from './convert';
import type { AcpStopReason } from './types';

/**
 * Build an ACP `session/update` notification with an
 * `agent_message_chunk` payload from an `assistant.delta` event.
 */
export function assistantDeltaToSessionUpdate(
  sessionId: string,
  event: AssistantDeltaEvent,
  messageId?: string,
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'agent_message_chunk',
      messageId,
      content: { type: 'text', text: event.delta },
    },
  };
}

/**
 * Map a {@link TurnEndReason} to an ACP `stopReason`.
 *
 * `completed` → `end_turn`: the model finished a clean turn.
 * `cancelled` → `cancelled`: the client/agent cancelled mid-turn.
 * `failed` throws a JSON-RPC error: `auth_required` for authentication
 *   failures, `internalError` otherwise. ACP has no failed stop reason;
 *   callers must reject the prompt request with this error.
 * `failed` + `provider.filtered` → `refusal`: the provider's safety policy
 *   blocked the response.
 * `blocked`   → `refusal`: a prompt hook blocked the turn before the model
 *   ran. ACP has no separate hook-blocked terminal state, so reuse the
 *   refusal channel.
 */
export function turnEndReasonToStopReason(
  reason: TurnEndReason,
  error?: { readonly code: string; readonly message?: string },
): AcpStopReason {
  switch (reason) {
    case 'completed':
      return 'end_turn';
    case 'cancelled':
      return 'cancelled';
    case 'failed':
      if (error?.code === 'provider.filtered') return 'refusal';
      if (isAuthError(error)) throw RequestError.authRequired(undefined, error?.message);
      throw RequestError.internalError(
        { code: error?.code },
        error?.message ?? 'The agent turn failed.',
      );
    case 'blocked':
      return 'refusal';
  }
}

/** Error codes that indicate an authentication / authorization failure. */
const AUTH_ERROR_CODES: ReadonlySet<string> = new Set([
  'provider.auth_error',
  'auth.login_required',
  'auth.token_missing',
  'auth.token_unauthorized',
  'auth.provisioning_required',
  'auth.model_not_resolved',
]);

/**
 * Whether the given error (from a `turn.ended` event) is an auth failure that
 * should surface as a JSON-RPC `auth_required` error so the ACP client
 * triggers its re-auth flow.
 */
export function isAuthError(error?: { readonly code: string }): boolean {
  return error !== undefined && AUTH_ERROR_CODES.has(error.code);
}

/**
 * Build the ACP `toolCallId` for a wire-level tool call.
 *
 * Composes `${turnId}:${toolCallId}` so multiple turns within a single
 * session (which legitimately reuse the same model-assigned tool call id
 * when the model retries) do not collide on the ACP side. The raw
 * `toolCallId` remains the in-process accumulator key — only the ACP wire
 * id is prefixed.
 */
export function acpToolCallId(turnId: number, toolCallId: string): string {
  return `${turnId}:${toolCallId}`;
}

/**
 * Heuristic map from a Kimi tool's `name` to ACP {@link ToolKind}.
 *
 * Pure, never throws — defaults to `'other'` whenever the name is
 * unrecognized so we never block streaming on an unknown tool.
 */
export function inferToolKind(name: string): ToolKind {
  switch (name) {
    case 'Read':
      return 'read';
    case 'Glob':
    case 'Grep':
      return 'search';
    case 'Write':
    case 'Edit':
      return 'edit';
    case 'Bash':
    case 'Terminal':
      return 'execute';
    case 'WebFetch':
    case 'WebSearch':
      return 'fetch';
    case 'Think':
      return 'think';
    default:
      return 'other';
  }
}

/**
 * Tools with a structured detail view on the client: an args JSON preview
 * only leaks into the unifiedDiff / output / content slots there (the daemon
 * maps text content into them), so it is suppressed. Unknown ('other') tools
 * keep the preview — it is their only visible content.
 */
function suppressArgsPreview(name: string | undefined): boolean {
  return name !== undefined && inferToolKind(name) !== 'other';
}

/**
 * Best-effort JSON stringification for tool args. Never throws — a streaming
 * push must never crash the prompt loop.
 */
export function stringifyArgs(args: unknown): string {
  try {
    return JSON.stringify(args) ?? String(args);
  } catch {
    return String(args);
  }
}

/**
 * File tools whose raw args carry a path worth advertising as a location.
 * v2 tools use `path`; `file_path` is accepted too for legacy-style args.
 */
const FILE_TOOL_NAMES: ReadonlySet<string> = new Set(['Read', 'Write', 'Edit', 'Glob', 'Grep']);

function argsPath(name: string, args: unknown): string | undefined {
  if (!FILE_TOOL_NAMES.has(name) || typeof args !== 'object' || args === null) return undefined;
  const record = args as Record<string, unknown>;
  for (const key of ['file_path', 'path']) {
    const value = record[key];
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return undefined;
}

/**
 * Derive the ACP {@link ToolCallLocation}s for a tool call, best-effort.
 *
 * Priority: the display block's path (diff / file_io — the same display data
 * `displayBlockToAcpContent` reads), then the raw args of the known file
 * tools. Only absolute paths are advertised (the wire contract requires
 * them); when nothing qualifies the caller omits the field rather than
 * fabricating one. No line information exists in either source, so `line`
 * stays unset.
 */
export function toolCallLocations(
  name: string,
  args: unknown,
  display: ToolInputDisplay | undefined,
): ToolCallLocation[] | undefined {
  // Search tools (Grep/Glob): the location is the search root, not a file the
  // call produces — clients render it as a bare path line that reads as
  // spilled output. Edit/Write/Read keep theirs (a real file link).
  if (inferToolKind(name) === 'search') return undefined;
  const displayPath =
    display !== undefined && (display.kind === 'diff' || display.kind === 'file_io')
      ? display.path
      : undefined;
  const path = [displayPath, argsPath(name, args)].find(
    (candidate): candidate is string => candidate !== undefined && isAbsolute(candidate),
  );
  if (path === undefined) return undefined;
  return [{ path }];
}

/**
 * Build the ACP `session/update` for the **initial** `tool_call` create
 * notification from a `tool.call.started` event.
 */
export function toolCallStartToSessionUpdate(
  sessionId: string,
  event: ToolCallStartedEvent,
): SessionNotification {
  const title = event.description ?? event.name;
  // A display block (diff/plan) replaces the textual args preview entirely:
  // the daemon maps any edit-kind text content to unifiedDiff, where it
  // overrides the structured diff and renders as raw JSON.
  const displayContent = event.display ? displayBlockToAcpContent(event.display) : null;
  const content: ToolCallContent[] = displayContent !== null ? [displayContent]
    : suppressArgsPreview(event.name) ? [] : [
      {
        type: 'content',
        content: { type: 'text', text: stringifyArgs(event.args) },
      },
    ];
  return {
    sessionId,
    update: {
      sessionUpdate: 'tool_call',
      toolCallId: acpToolCallId(event.turnId, event.toolCallId),
      title,
      kind: inferToolKind(event.name),
      status: 'in_progress',
      rawInput: event.args,
      locations: toolCallLocations(event.name, event.args, event.display),
      content,
    },
  };
}

/**
 * Build a `tool_call_update` for a streaming arguments delta. Mutates
 * `accumulator.args` with the new fragment and emits cumulative REPLACE
 * content.
 */
export function toolCallDeltaToSessionUpdate(
  sessionId: string,
  event: ToolCallDeltaEvent,
  accumulator: { args: string; suppressArgs?: boolean },
): SessionNotification {
  accumulator.args += event.argumentsPart ?? '';
  accumulator.suppressArgs = accumulator.suppressArgs || suppressArgsPreview(event.name);
  // The preview text stays suppressed for structured kinds, but rawInput is
  // data, not display: forward it as soon as the args JSON parses so clients
  // can show the command even before tool.call.started — or after a cancel
  // that lands mid-stream. Edit/Write additionally get a live diff block so
  // the original→new comparison renders during streaming.
  let rawInput: unknown;
  try { rawInput = JSON.parse(accumulator.args); } catch { rawInput = undefined; }
  const diff = rawInput !== undefined ? editDiffFromArgs(event.name, rawInput) : null;
  const content: ToolCallContent[] = diff !== null ? [diff]
    : accumulator.suppressArgs ? [] : [
      {
        type: 'content',
        content: { type: 'text', text: accumulator.args },
      },
    ];
  return {
    sessionId,
    update: {
      sessionUpdate: 'tool_call_update',
      toolCallId: acpToolCallId(event.turnId, event.toolCallId),
      status: 'in_progress',
      // Live generation feedback rides the title — the one slot clients never
      // reinterpret as output/diff. Once args parse, structured data takes over.
      ...(rawInput === undefined ? { title: argsPreviewTitle(accumulator.args) } : {}),
      ...(rawInput !== undefined ? { rawInput } : {}),
      content,
    },
  };
}

/** Compact tail of the streaming args for the live title preview. */
function argsPreviewTitle(args: string): string {
  const compact = args.replaceAll(/\s+/g, ' ').trim();
  const tail = compact.length > 40 ? `…${compact.slice(-40)}` : compact;
  return `Generating args… ${tail}`;
}

/** Build an inline diff from parsed Edit/Write args; null for anything else. */
function editDiffFromArgs(name: string | undefined, args: unknown): ToolCallContent | null {
  if (inferToolKind(name ?? '') !== 'edit' || args === null || typeof args !== 'object') return null;
  const record = args as { path?: unknown; old_string?: unknown; new_string?: unknown; content?: unknown };
  if (typeof record.path !== 'string') return null;
  if (typeof record.old_string === 'string' && typeof record.new_string === 'string') {
    return { type: 'diff', path: record.path, oldText: record.old_string, newText: record.new_string };
  }
  if (typeof record.content === 'string') {
    return { type: 'diff', path: record.path, oldText: null, newText: record.content };
  }
  return null;
}

/**
 * Build the initial ACP `tool_call` (CREATE) notification from the **first**
 * `tool.call.delta` event for a given `toolCallId`.
 *
 * agent-core-v2 emits `tool.call.delta` events while the provider streams the
 * model's tool-call args, and only later emits `tool.call.started` (after the
 * streaming phase, when the call is dispatched). Lazy-creating the wire
 * tool_call from the first delta gives subsequent deltas a legitimate parent
 * to update, so the client never sees an update before its create.
 */
export function toolCallLazyCreateToSessionUpdate(
  sessionId: string,
  event: ToolCallDeltaEvent,
): SessionNotification {
  const name = event.name ?? 'tool';
  return {
    sessionId,
    update: {
      sessionUpdate: 'tool_call',
      toolCallId: acpToolCallId(event.turnId, event.toolCallId),
      title: name,
      kind: event.name ? inferToolKind(event.name) : 'other',
      status: 'pending',
      content: suppressArgsPreview(event.name) ? [] : [
        {
          type: 'content',
          content: { type: 'text', text: event.argumentsPart ?? '' },
        },
      ],
    },
  };
}

/**
 * Build a `tool_call_update` that finalises a lazy-created tool call once
 * `tool.call.started` arrives. Used only when
 * {@link toolCallLazyCreateToSessionUpdate} already emitted a `tool_call` for
 * this `toolCallId` from a streaming delta — we cannot send a second CREATE,
 * so the canonical metadata is delivered as an update instead.
 */
export function toolCallStartedUpgradeToSessionUpdate(
  sessionId: string,
  event: ToolCallStartedEvent,
): SessionNotification {
  const title = event.description ?? event.name;
  const displayContent = event.display ? displayBlockToAcpContent(event.display) : null;
  const content: ToolCallContent[] = displayContent !== null ? [displayContent]
    : suppressArgsPreview(event.name) ? [] : [
      {
        type: 'content',
        content: { type: 'text', text: stringifyArgs(event.args) },
      },
    ];
  return {
    sessionId,
    update: {
      sessionUpdate: 'tool_call_update',
      toolCallId: acpToolCallId(event.turnId, event.toolCallId),
      title,
      kind: inferToolKind(event.name),
      status: 'in_progress',
      rawInput: event.args,
      locations: toolCallLocations(event.name, event.args, event.display),
      content,
    },
  };
}

/**
 * Map a `tool.progress` event to an ACP `tool_call_update`. Only
 * `update.kind === 'status'` with non-empty `text` produces a notification
 * (refreshes the tool card title); everything else returns `null`.
 */
export function toolProgressToSessionUpdate(
  sessionId: string,
  event: ToolProgressEvent,
): SessionNotification | null {
  if (event.update.kind === 'status' && event.update.text) {
    return {
      sessionId,
      update: {
        sessionUpdate: 'tool_call_update',
        toolCallId: acpToolCallId(event.turnId, event.toolCallId),
        title: event.update.text,
      },
    };
  }
  return null;
}

/**
 * Map a `thinking.delta` event to an `agent_thought_chunk` notification.
 */
export function thinkingDeltaToSessionUpdate(
  sessionId: string,
  event: ThinkingDeltaEvent,
  messageId?: string,
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'agent_thought_chunk',
      messageId,
      content: { type: 'text', text: event.delta },
    },
  };
}

/**
 * Map a `tool.result` event to the **terminal** `tool_call_update`
 * notification for that call. `status` flips to `completed` (success) or
 * `failed` (`event.isError === true`); content replaces the streaming args
 * preview with the final tool output; `rawOutput` preserves the raw output.
 * `ToolResultEvent` carries no args/display, so `locations` (derived at
 * `tool.call.started` by the caller) is re-attached here when available.
 */
export function toolResultToSessionUpdate(
  sessionId: string,
  event: ToolResultEvent,
  locations?: ToolCallLocation[],
  display?: ToolInputDisplay,
  title?: string,
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'tool_call_update',
      toolCallId: acpToolCallId(event.turnId, event.toolCallId),
      status: event.isError ? 'failed' : 'completed',
      title,
      content: withToolDisplay(toolResultToAcpContent(event), display, event.isError),
      rawOutput: event.output,
      locations,
    },
  };
}

/**
 * Translate a TodoList display block into an ACP `plan` session update.
 * `done` rewrites to `completed`; `priority` defaults to `'medium'`. Empty
 * entries clear the client's previous plan.
 */
export function todoListToSessionUpdate(
  sessionId: string,
  turnId: number,
  items: ReadonlyArray<{ title: string; status: string }>,
): SessionNotification {
  void turnId;
  const entries: PlanEntry[] = items.map((item) => ({
    content: item.title,
    priority: 'medium',
    status: mapTodoStatus(item.status),
  }));
  return {
    sessionId,
    update: {
      sessionUpdate: 'plan',
      entries,
    },
  };
}

function mapTodoStatus(status: string): PlanEntryStatus {
  switch (status) {
    case 'pending':
      return 'pending';
    case 'in_progress':
      return 'in_progress';
    case 'done':
    case 'completed':
      return 'completed';
    default:
      return 'pending';
  }
}

/**
 * If the given {@link ToolInputDisplay} carries a TodoList payload, project it
 * into an ACP `plan` session update. Returns `null` for every other display
 * kind.
 */
export function planFromDisplayBlock(
  sessionId: string,
  turnId: number,
  display: ToolInputDisplay,
): SessionNotification | null {
  if (display.kind !== 'todo_list') return null;
  return todoListToSessionUpdate(sessionId, turnId, display.items);
}

/**
 * Build a one-shot ACP `available_commands_update` session notification.
 */
export function availableCommandsUpdateNotification(
  sessionId: string,
  commands: ReadonlyArray<AvailableCommand> = [],
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'available_commands_update',
      availableCommands: commands.slice(),
    },
  };
}

/**
 * Build a `current_mode_update` session notification, emitted after
 * `session/set_mode` (or the `mode` config-option arm) changes the active
 * mode. Coexists with `config_option_update`: the two serve clients reading
 * the first-class `modes` state and clients reading `configOptions`
 * respectively.
 */
export function currentModeUpdateNotification(
  sessionId: string,
  currentModeId: string,
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'current_mode_update',
      currentModeId,
    },
  };
}

/**
 * Build a `config_option_update` session notification, emitted after the model
 * / mode / thinking pickers change so clients repaint the dropdown's selected
 * indicator.
 */
export function configOptionUpdateNotification(
  sessionId: string,
  configOptions: readonly SessionConfigOption[],
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'config_option_update',
      configOptions: [...configOptions],
    },
  };
}

/**
 * Build a one-shot `usage_update` session notification, emitted after a turn
 * settles. `used` is the agent's current context token count, `size` the bound
 * model's max context size; `cost` stays omitted (the engine has no cost
 * data).
 */
export function usageUpdateNotification(
  sessionId: string,
  used: number,
  size: number,
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'usage_update',
      used,
      size,
    },
  };
}

/**
 * Build a `session_info_update` session notification for a title change.
 * `title: null` clears the title client-side.
 */
export function sessionInfoUpdateNotification(
  sessionId: string,
  title: string | null,
): SessionNotification {
  return {
    sessionId,
    update: {
      sessionUpdate: 'session_info_update',
      title,
    },
  };
}

/** ACP replaces content on update; preserve the tool's display through completion. */
export function withToolDisplay(
  content: ToolCallContent[],
  display?: ToolInputDisplay,
  isError?: boolean,
): ToolCallContent[] {
  const block = display === undefined ? null : displayBlockToAcpContent(display);
  // A successful call with a structured diff carries the diff only: the
  // textual output duplicates it (rawOutput preserves the full result), and
  // some clients render text content in place of the diff. Failures keep
  // their error text.
  if (block !== null && isError !== true) return [block];
  return block === null ? content : [block, ...content];
}

/** Both observed context messages and replayed messages cross this boundary. */
export function contextMessageToSessionUpdates(
  sessionId: string,
  message: ContextMessage,
  fallbackId: string,
): SessionNotification[] {
  if (message.role !== 'user' && message.role !== 'system') return [];
  const origin = message.origin;
  if (message.role === 'system' && origin === undefined) return [];
  const text = message.content.filter(p => p.type === 'text').map(p => p.text).join('');
  if (!text) return [];
  return [{
    sessionId,
    update: {
      sessionUpdate: 'user_message_chunk',
      messageId: message.id ?? fallbackId,
      content: { type: 'text', text: formatContextNotification(text, origin) },
      _meta: { origin },
    },
  }];
}

/** Keep established envelopes intact; wrap only otherwise unmarked context. */
export function formatContextNotification(text: string, origin: ContextMessage['origin']): string {
  if (origin?.kind === 'shell_command') return text;
  if (origin === undefined || origin.kind === 'user') return text;
  const trimmed = text.trim();
  const envelope = /^<([a-z][a-z0-9-]*)(?:\s[^<>]*)?>[\s\S]*<\/\1>$/i;
  if (envelope.test(trimmed)) return text;
  if (/<skill-loaded(?:\s[^<>]*)?>[\s\S]*<\/skill-loaded>/.test(text)) return text;
  let category: string;
  let type: string;
  let title: string;
  switch (origin.kind) {
    case 'task':
      category = 'task'; type = `task.${origin.status}`; title = `Background task ${origin.status}`; break;
    case 'cron_job':
      category = 'cron'; type = 'cron.fired'; title = 'Scheduled task'; break;
    case 'cron_missed':
      category = 'cron'; type = 'cron.missed'; title = 'Missed scheduled tasks'; break;
    case 'system_trigger':
      category = origin.name.startsWith('goal_') ? 'goal' : 'system';
      type = category === 'goal' ? `goal.${origin.name.slice('goal_'.length)}` : `system.${origin.name}`;
      title = origin.name === 'goal_continuation' ? 'Continue active goal' : origin.name; break;
    case 'injection':
      category = 'context'; type = `context.${origin.variant}`; title = origin.variant; break;
    case 'compaction_summary':
      category = 'context'; type = 'context.compaction'; title = 'Conversation summary'; break;
    case 'hook_result':
      category = 'hook'; type = `hook.${origin.event}`; title = origin.event; break;
    case 'skill_activation':
      category = 'skill'; type = 'skill.activation'; title = origin.skillName; break;
    case 'plugin_command':
      category = 'command'; type = 'command.activation'; title = `${origin.pluginId}:${origin.commandName}`; break;
    case 'retry':
      category = 'system'; type = 'system.retry'; title = 'Retry'; break;
    default:
      return text;
  }
  return `<notification category="${escapeNotificationXml(category)}" type="${escapeNotificationXml(type)}">\nTitle: ${escapeNotificationXml(title)}\n\n${escapeNotificationXml(text)}\n</notification>`;
}

function escapeNotificationXml(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&apos;');
}

export function legacyMessageId(message: ContextMessage, position: number): string {
  return `context:${position}:${createHash('sha256').update(JSON.stringify(message)).digest('hex').slice(0, 16)}`;
}

/** One terminal tool projection, shared by live delivery and journal replay. */
export function completedToolToSessionUpdate(
  sessionId: string,
  event: ToolResultEvent,
  call?: ToolCallStartedEvent,
): SessionNotification {
  return toolResultToSessionUpdate(
    sessionId, event,
    call === undefined ? undefined : toolCallLocations(call.name, call.args, call.display),
    call?.display,
    call === undefined ? undefined : call.description ?? call.name,
  );
}
