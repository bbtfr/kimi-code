import { describe, expect, it } from 'vitest';

import { contextMessageToSessionUpdates, formatContextNotification } from '../src/events-map';
import { projectHistoryToSessionUpdates } from '../src/replay';

import type { SessionNotification } from '@agentclientprotocol/sdk';
import type { ContextMessage } from '@moonshot-ai/agent-core-v2';

const SESSION_ID = 'session_test';

function kinds(updates: readonly SessionNotification[]): string[] {
  return updates.map((u) => u.update.sessionUpdate);
}

describe('projectHistoryToSessionUpdates', () => {
  it('returns an empty array for an empty history', () => {
    expect(projectHistoryToSessionUpdates(SESSION_ID, [])).toEqual([]);
  });

  it('projects a user text message to a user_message_chunk', () => {
    const messages: ContextMessage[] = [
      { role: 'user', content: [{ type: 'text', text: 'hi' }], toolCalls: [] },
    ];
    const updates = projectHistoryToSessionUpdates(SESSION_ID, messages);
    expect(kinds(updates)).toEqual(['user_message_chunk']);
    expect(updates[0]?.update).toMatchObject({
      sessionUpdate: 'user_message_chunk',
      content: { type: 'text', text: 'hi' },
    });
  });

  it('projects an assistant text + tool call and correlates the tool result', () => {
    const messages: ContextMessage[] = [
      { role: 'user', content: [{ type: 'text', text: 'read a.ts' }], toolCalls: [] },
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'reading' }],
        toolCalls: [{ type: 'function', id: 'c1', name: 'Read', arguments: '{"path":"a.ts"}' }],
      },
      {
        role: 'tool',
        content: [{ type: 'text', text: 'file body' }],
        toolCalls: [],
        toolCallId: 'c1',
      },
    ];
    const updates = projectHistoryToSessionUpdates(SESSION_ID, messages);
    expect(kinds(updates)).toEqual([
      'user_message_chunk',
      'agent_message_chunk',
      'tool_call',
      'tool_call_update',
    ]);
    const create = updates[2]?.update;
    expect(create).toMatchObject({
      sessionUpdate: 'tool_call',
      toolCallId: '1:c1',
      status: 'in_progress',
    });
    const done = updates[3]?.update;
    expect(done).toMatchObject({
      sessionUpdate: 'tool_call_update',
      toolCallId: '1:c1',
      status: 'completed',
    });
  });

  it('marks an errored tool result as failed', () => {
    const messages: ContextMessage[] = [
      {
        role: 'assistant',
        content: [],
        toolCalls: [{ type: 'function', id: 'c1', name: 'Bash', arguments: '{}' }],
      },
      {
        role: 'tool',
        content: [{ type: 'text', text: 'boom' }],
        toolCalls: [],
        toolCallId: 'c1',
        isError: true,
      },
    ];
    const updates = projectHistoryToSessionUpdates(SESSION_ID, messages);
    expect(updates.at(-1)?.update).toMatchObject({ status: 'failed' });
  });

  it('projects a think part to an agent_thought_chunk', () => {
    const messages: ContextMessage[] = [
      {
        role: 'assistant',
        content: [{ type: 'think', think: 'hmm' }],
        toolCalls: [],
      },
    ];
    const updates = projectHistoryToSessionUpdates(SESSION_ID, messages);
    expect(kinds(updates)).toEqual(['agent_thought_chunk']);
  });

  it('skips a tool message whose call was never issued in this slice', () => {
    const messages: ContextMessage[] = [
      {
        role: 'tool',
        content: [{ type: 'text', text: 'orphan' }],
        toolCalls: [],
        toolCallId: 'unknown',
      },
    ];
    expect(projectHistoryToSessionUpdates(SESSION_ID, messages)).toEqual([]);
  });

  it('increments the synthetic turnId per assistant message', () => {
    const messages: ContextMessage[] = [
      {
        role: 'assistant',
        content: [],
        toolCalls: [{ type: 'function', id: 'a', name: 'Read', arguments: '{}' }],
      },
      {
        role: 'assistant',
        content: [],
        toolCalls: [{ type: 'function', id: 'b', name: 'Read', arguments: '{}' }],
      },
    ];
    const updates = projectHistoryToSessionUpdates(SESSION_ID, messages);
    const ids = updates
      .filter((u) => u.update.sessionUpdate === 'tool_call')
      .map((u) => (u.update as { toolCallId: string }).toolCallId);
    expect(ids).toEqual(['1:a', '2:b']);
  });
});


describe('context notification envelopes', () => {
  it('preserves existing envelopes and tagged skill instructions byte for byte', () => {
    for (const text of ['<notification type="task.completed">Title: Done\na < b & c</notification>', '<cron-fire jobId="1"><prompt>a & b</prompt></cron-fire>', '<system-reminder>Today</system-reminder>', 'Skill loaded.\n<skill-loaded name="example">Instructions</skill-loaded>']) {
      expect(formatContextNotification(text, { kind: 'injection', variant: 'example' })).toBe(text);
    }
  });
  it('does not wrap user or shell content', () => {
    for (const text of ['hello & goodbye', '<bash-stdout>ok</bash-stdout><bash-stderr></bash-stderr>']) {
      expect(formatContextNotification(text, { kind: 'user' })).toBe(text);
      expect(formatContextNotification(text, { kind: 'shell_command', phase: 'output' })).toBe(text);
      expect(formatContextNotification(text, undefined)).toBe(text);
    }
  });
  it('escapes bare notifications without mutating their persisted messages', () => {
    const message: ContextMessage = { role: 'user', content: [{ type: 'text', text: 'a < b & </notification>' }], toolCalls: [], origin: { kind: 'system_trigger', name: 'goal_continuation' } };
    const before = JSON.stringify(message);
    const expected = '<notification category="goal" type="goal.continuation">\nTitle: Continue active goal\n\na &lt; b &amp; &lt;/notification&gt;\n</notification>';
    expect(formatContextNotification(message.content[0]!.type === 'text' ? message.content[0]!.text : '', message.origin)).toBe(expected);
    expect(projectHistoryToSessionUpdates(SESSION_ID, [message])[0]?.update).toMatchObject({ content: { type: 'text', text: expected } });
    expect(JSON.stringify(message)).toBe(before);
  });
  it('wraps unmarked summaries, plugin commands, hooks, missed cron and retries', () => {
    const origins: NonNullable<ContextMessage['origin']>[] = [
      { kind: 'compaction_summary' }, { kind: 'cron_missed', count: 1 },
      { kind: 'plugin_command', activationId: 'a', pluginId: 'example', commandName: 'check', trigger: 'user-slash' },
      { kind: 'hook_result', event: 'Stop' }, { kind: 'retry' },
    ];
    for (const origin of origins) expect(formatContextNotification('original body', origin)).toMatch(/^<notification .*original body\n<\/notification>$/s);
  });
});


describe('opaque client envelopes', () => {
  it('preserves client text and user channel in live context conversion and history', () => {
    const text = '<client-context>\nClient-specific input & <details>\n</client-context>';
    const message: ContextMessage = { id: 'client-message', role: 'user', origin: { kind: 'user' }, content: [{ type: 'text', text }], toolCalls: [] };
    const live = contextMessageToSessionUpdates(SESSION_ID, message, 'fallback');
    expect(projectHistoryToSessionUpdates(SESSION_ID, [message])).toEqual(live);
    expect(live[0]?.update).toMatchObject({ sessionUpdate: 'user_message_chunk', content: { type: 'text', text } });
    expect(formatContextNotification(text, { kind: 'injection', variant: 'test' })).toBe(text);
  });
});
