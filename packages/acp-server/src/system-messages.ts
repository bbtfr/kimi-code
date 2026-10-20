import { randomUUID } from 'node:crypto';

import type { ContentBlock, SessionNotification } from '@agentclientprotocol/sdk';

type NotificationUpdate = SessionNotification['update'];

const escapeXml = (text: string): string =>
  text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');

/**
 * Legacy `<paseo-system>` prompt → display envelope; anything else passes
 * through as plain text. Model input is never touched by this module — the
 * projection only rewrites what the client displays.
 */
export function projectSystemMessage(text: string): string | undefined {
  const envelope = /^<paseo-system>\n([\s\S]*)\n<\/paseo-system>$/.exec(text);
  if (!envelope) return undefined;
  const body = envelope[1]!;
  const schedule = /^Schedule(?: "([^\n]*)")? fired \(id=([^\s,()]+), run=([^\s,()]+)\)\.\n([\s\S]+)$/.exec(body);
  if (!schedule) {
    return `<notification category="system" type="paseo.system" source="paseo">\nTitle: System notification\n\n${escapeXml(body)}\n</notification>`;
  }
  const name = schedule[1] === undefined ? '' : ` name="${escapeXml(schedule[1])}"`;
  return `<cron-fire source="paseo" jobId="${escapeXml(schedule[2]!)}" runId="${escapeXml(schedule[3]!)}"${name}>\n<prompt>${escapeXml(body)}</prompt>\n</cron-fire>`;
}

const executionUpdates = new Set(['agent_message_chunk', 'agent_thought_chunk', 'tool_call', 'tool_call_update']);
const prefix = '<paseo-system>\n';

interface PendingProjection {
  text: string;
  projected: string;
  shown: boolean;
  messageId: string;
}

interface FragmentBuffer {
  text: string;
  messageId?: string;
  messages: SessionNotification[];
}

/**
 * Per-session display projector for legacy `<paseo-system>` inputs and edit
 * results — the in-process replacement for the former stdio proxy layer.
 * Registration (`onPrompt`/`onPromptSettled`) covers the prompt lifecycle;
 * `transform` rewrites live emissions; `transformReplay` covers cold loads.
 */
export class SystemMessageProjector {
  private readonly pendings = new Map<string, PendingProjection>();
  private fragments?: FragmentBuffer;
  private promptSeq = 0;

  constructor(private readonly sessionId: string) {}

  /** Returns a key for `onPromptSettled` when this prompt needs a mirror. */
  onPrompt(blocks: readonly ContentBlock[]): string | undefined {
    if (blocks.length !== 1 || blocks[0]?.type !== 'text') return undefined;
    const text = blocks[0].text;
    if (typeof text !== 'string') return undefined;
    const projected = projectSystemMessage(text);
    if (projected === undefined) return undefined;
    const key = `prompt-${++this.promptSeq}`;
    this.pendings.set(key, { text, projected, shown: false, messageId: `paseo-system:${randomUUID()}` });
    return key;
  }

  /**
   * Live emission path: mirror the pending projection before the turn's first
   * execution update, rewrite user echoes of legacy envelopes, and buffer
   * prefix fragments split across chunks. The returned notifications are
   * already projected — emit them as-is, in order.
   */
  transform(notification: SessionNotification): SessionNotification[] {
    const update = notification.update;
    const current = notification;
    const pending = this.firstPending();

    if (update.sessionUpdate === 'user_message_chunk' && update.content?.type === 'text') {
      const out: SessionNotification[] = [];
      if (this.fragments && this.fragments.messageId !== update.messageId) {
        out.push(...this.flush());
      }
      const text = (this.fragments?.text ?? '') + update.content.text;
      const projected = projectSystemMessage(text);
      if (projected !== undefined) {
        this.fragments = undefined;
        if (pending?.text === text) {
          if (pending.shown) return out;
          pending.shown = true;
        }
        return [...out, this.asAgentChunk(current, update, projected)];
      }
      if (prefix.startsWith(text) || (text.startsWith(prefix) && !text.includes('</paseo-system>'))) {
        this.fragments = {
          text,
          messageId: update.messageId ?? undefined,
          messages: [...(this.fragments?.messages ?? []), current],
        };
        return out;
      }
      return [...out, ...this.flush(), current];
    }

    const out = this.flush();
    // An older fork in a rolling upgrade may already emit this projection.
    if (
      pending && update.sessionUpdate === 'agent_message_chunk'
      && update.content?.type === 'text' && update.content.text === pending.projected
    ) {
      if (pending.shown) return out;
      pending.shown = true;
    } else if (pending && executionUpdates.has(update.sessionUpdate)) {
      out.push(...this.mirror(pending));
    }
    return [...out, current];
  }

  /** Cold replay: complete messages convert directly; no prompt machinery. */
  transformReplay(notification: SessionNotification): SessionNotification {
    const originalUpdate = notification.update;
    let update = originalUpdate;
    if (update.sessionUpdate === 'user_message_chunk' && update.content?.type === 'text') {
      const projected = projectSystemMessage(update.content.text);
      if (projected !== undefined) {
        update = { ...update, sessionUpdate: 'agent_message_chunk', content: { ...update.content, text: projected } };
      }
    }
    return update === originalUpdate ? notification : { ...notification, update };
  }

  /** Prompt settled: an empty turn still displays its input via the mirror. */
  onPromptSettled(key: string | undefined, hasResult: boolean): SessionNotification[] {
    const out = this.flush();
    if (key === undefined) return out;
    const pending = this.pendings.get(key);
    this.pendings.delete(key);
    if (pending && hasResult) out.push(...this.mirror(pending));
    return out;
  }

  /** Session teardown: re-emit any buffered raw fragments, never swallow them. */
  end(): SessionNotification[] {
    const out = this.flush();
    this.pendings.clear();
    return out;
  }

  private firstPending(): PendingProjection | undefined {
    return this.pendings.values().next().value;
  }

  private mirror(pending: PendingProjection): SessionNotification[] {
    if (pending.shown) return [];
    pending.shown = true;
    return [{
      sessionId: this.sessionId,
      update: {
        sessionUpdate: 'agent_message_chunk',
        messageId: pending.messageId,
        content: { type: 'text', text: pending.projected },
      },
    }];
  }

  private flush(): SessionNotification[] {
    const buffer = this.fragments;
    if (!buffer) return [];
    this.fragments = undefined;
    return buffer.messages;
  }

  private asAgentChunk(
    notification: SessionNotification,
    update: NotificationUpdate,
    projected: string,
  ): SessionNotification {
    return {
      ...notification,
      update: {
        ...update,
        sessionUpdate: 'agent_message_chunk',
        content: { ...(update as { content?: object }).content, text: projected },
      } as NotificationUpdate,
    };
  }
}
