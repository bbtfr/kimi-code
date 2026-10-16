import assert from 'node:assert/strict';

import type { ContentBlock, SessionNotification } from '@agentclientprotocol/sdk';
import { describe, expect, it } from 'vitest';

import { projectEditResult, projectSystemMessage, SystemMessageProjector } from '../src/system-messages';

const raw = '<paseo-system>\nSchedule "Check & report" fired (id=s1, run=r1).\nRead <file> & report.\n</paseo-system>';
const projected = projectSystemMessage(raw)!;

const promptBlocks = (text = raw): ContentBlock[] => [{ type: 'text', text }];
const update = (
  sessionUpdate: string,
  text: string,
  messageId = 'native-1',
): SessionNotification => ({
  sessionId: 'session',
  update: { sessionUpdate, messageId, content: { type: 'text', text } } as SessionNotification['update'],
});
const toolResult = (fields: Record<string, unknown> = {}): SessionNotification => ({
  sessionId: 'session',
  update: {
    sessionUpdate: 'tool_call_update', toolCallId: '1:edit', status: 'completed',
    content: [
      { type: 'diff', path: '/example/file.txt', oldText: 'before\n', newText: 'after\n' },
      { type: 'content', content: { type: 'text', text: 'File updated' } },
    ],
    rawOutput: 'File updated',
    ...fields,
  } as unknown as SessionNotification['update'],
});

describe('projectEditResult', () => {
  it('successful structured edits retain diff and raw output without a duplicate text override', () => {
    for (const rawOutput of ['File updated', { message: 'File updated' }, [{ type: 'text', text: 'File updated' }]]) {
      const text = typeof rawOutput === 'string' ? rawOutput : JSON.stringify(rawOutput);
      const input = toolResult({ rawOutput, content: [
        { type: 'diff', path: '/example/file.txt', oldText: 'before\n', newText: 'after\n' },
        { type: 'content', content: { type: 'text', text } },
      ] });
      const before = structuredClone(input);
      const update = projectEditResult(input.update);
      assert.deepEqual((update as { content?: unknown[] }).content, [{ type: 'diff', path: '/example/file.txt', oldText: 'before\n', newText: 'after\n' }]);
      assert.deepEqual(input, before, 'the original payload must not be mutated');
      assert.deepEqual(projectEditResult(update), update, 'projection is idempotent');
    }
  });

  it('leaves failures, pending tools and nonredundant content untouched', () => {
    for (const fields of [
      { status: 'failed' }, { status: 'in_progress' }, { status: 'pending' },
      { kind: 'execute' }, { kind: 'read' }, { rawOutput: undefined },
      { rawOutput: 'Different output' },
      { content: [{ type: 'content', content: { type: 'text', text: 'File updated' } }] },
    ]) {
      const input = toolResult(fields);
      assert.equal(projectEditResult(input.update), input.update);
    }
  });
});

describe('SystemMessageProjector', () => {
  it('live non-echoed input mirrors before the first execution update; replay converts the echo identically', () => {
    const projector = new SystemMessageProjector('session');
    const key = projector.onPrompt(promptBlocks());
    assert.ok(key);
    const answer = update('agent_message_chunk', 'done', 'answer');
    const live = projector.transform(answer);
    assert.equal(live.length, 2);
    assert.equal(live[0]!.update.sessionUpdate, 'agent_message_chunk');
    assert.equal((live[0]!.update as { content?: { text?: string } }).content?.text, projected);
    assert.match(projected, /name="Check &amp; report"/);
    assert.match(projected, /Read &lt;file&gt; &amp; report/);
    assert.deepEqual(live[1], answer);
    assert.deepEqual(projector.onPromptSettled(key, true), []);
    const replay = projector.transformReplay(update('user_message_chunk', raw));
    assert.equal((replay.update as { content?: { text?: string } }).content?.text, projected);
    assert.equal(replay.update.sessionUpdate, 'agent_message_chunk');
    assert.equal((replay.update as { messageId?: string }).messageId, 'native-1');
  });

  it('no duplicate when generic user echo or old-fork mirror arrives before/after model output', () => {
    for (const oldFork of [false, true]) for (const echoFirst of [false, true]) {
      const projector = new SystemMessageProjector('session');
      const key = projector.onPrompt(promptBlocks());
      const echo = oldFork ? update('agent_message_chunk', projected) : update('user_message_chunk', raw);
      const answer = update('agent_message_chunk', 'done', 'answer');
      const out = (echoFirst ? [echo, answer] : [answer, echo]).flatMap(m => projector.transform(m));
      out.push(...projector.onPromptSettled(key, true));
      assert.equal(
        out.filter(m => (m.update as { content?: { text?: string } }).content?.text === projected).length,
        1, `oldFork=${oldFork} echoFirst=${echoFirst}`,
      );
    }
  });

  it('failed submission shows no mirror and clears the pending projection', () => {
    const projector = new SystemMessageProjector('session');
    const key = projector.onPrompt(promptBlocks());
    assert.deepEqual(projector.onPromptSettled(key, false), []);
    const out = projector.transform(update('agent_message_chunk', 'done', 'answer'));
    assert.equal(out.length, 1);
  });

  it('successful empty turn still displays input; repeated identical inputs remain separate', () => {
    const projector = new SystemMessageProjector('session');
    const ids: string[] = [];
    for (const _ of [1, 2]) {
      const key = projector.onPrompt(promptBlocks());
      const out = projector.onPromptSettled(key, true);
      assert.equal(out.length, 1);
      ids.push((out[0]!.update as { messageId?: string }).messageId!);
    }
    assert.notEqual(ids[0], ids[1]);
  });

  it('fragmented input converts once; malformed, versioned and ordinary messages pass through', () => {
    for (let split = 1; split < raw.length; split++) {
      const projector = new SystemMessageProjector('session');
      const out = [
        update('user_message_chunk', raw.slice(0, split), 'frag'),
        update('user_message_chunk', raw.slice(split), 'frag'),
      ].flatMap(m => projector.transform(m));
      assert.equal(out.length, 1, `split=${split}`);
      assert.equal((out[0]!.update as { content?: { text?: string } }).content?.text, projected);
    }
    for (const text of ['hello', 'Example: ' + raw, '<paseo-system version="1" kind="user-message">\nhello\n</paseo-system>', '<paseo-system>\nincomplete', '<bash-stdout>ok</bash-stdout>']) {
      const projector = new SystemMessageProjector('session');
      const message = update('user_message_chunk', text);
      assert.equal(projectSystemMessage(text), undefined);
      assert.deepEqual([...projector.transform(message), ...projector.end()], [message], `text=${text.slice(0, 30)}`);
    }
    assert.match(projectSystemMessage('<paseo-system>\nUnknown & event\n</paseo-system>')!, /type="paseo.system"/);
  });
});
