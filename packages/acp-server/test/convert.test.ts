import { mkdtemp, readdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { McpServer } from '@agentclientprotocol/sdk';
import type { ContentPart } from '@moonshot-ai/agent-core-v2';
import { afterEach, describe, expect, it } from 'vitest';

import {
  acpBlocksToContentParts,
  acpMcpServersToConfigRecord,
  compressPromptImageParts,
  displayBlockToAcpContent,
} from '../src/convert';
import {
  toolCallDeltaToSessionUpdate,
  toolCallLazyCreateToSessionUpdate,
  toolCallStartToSessionUpdate,
  toolCallStartedUpgradeToSessionUpdate,
  toolResultToSessionUpdate,
} from '../src/events-map';
import { solidPng, solidPngBase64 } from './_helpers/png';

describe('acpMcpServersToConfigRecord', () => {
  it('returns undefined for an absent or empty list', () => {
    expect(acpMcpServersToConfigRecord(undefined)).toBeUndefined();
    expect(acpMcpServersToConfigRecord([])).toBeUndefined();
  });

  it('maps stdio servers (no type field) to local stdio configs', () => {
    const servers: McpServer[] = [
      {
        name: 'fs',
        command: '/usr/local/bin/mcp-fs',
        args: ['--root', '/tmp'],
        env: [
          { name: 'API_KEY', value: 'secret' },
          { name: 'DEBUG', value: '1' },
        ],
      },
    ];
    expect(acpMcpServersToConfigRecord(servers)).toEqual({
      fs: {
        transport: 'stdio',
        command: '/usr/local/bin/mcp-fs',
        args: ['--root', '/tmp'],
        env: { API_KEY: 'secret', DEBUG: '1' },
        runtime_id: 'local',
      },
    });
  });

  it('maps http and sse servers with header pairs as a record', () => {
    const servers: McpServer[] = [
      {
        type: 'http',
        name: 'web',
        url: 'https://example.com/mcp',
        headers: [{ name: 'Authorization', value: 'Bearer x' }],
      },
      { type: 'sse', name: 'events', url: 'https://example.com/sse', headers: [] },
    ];
    expect(acpMcpServersToConfigRecord(servers)).toEqual({
      web: {
        transport: 'http',
        url: 'https://example.com/mcp',
        headers: { Authorization: 'Bearer x' },
      },
      events: { transport: 'sse', url: 'https://example.com/sse', headers: undefined },
    });
  });

  it('drops the unstable acp transport and returns undefined when nothing survives', () => {
    const servers = [{ type: 'acp', name: 'nested', serverId: 'srv-1' } as unknown as McpServer];
    expect(acpMcpServersToConfigRecord(servers)).toBeUndefined();
  });
});

describe('acpBlocksToContentParts', () => {
  it('projects file links and embedded text resources with provenance', () => {
    const parts = acpBlocksToContentParts([
      { type: 'resource_link', uri: 'file:///tmp/example.ts#L2-L4', name: 'example.ts' },
      { type: 'resource_link', uri: 'https://example.test/doc', name: 'remote doc' },
      {
        type: 'resource',
        resource: { uri: 'memory://note/1', text: 'remember this' },
      },
    ] as never);

    expect(parts).toEqual([
      { type: 'text', text: '/tmp/example.ts:2-4' },
      {
        type: 'text',
        text: '<resource_link uri="https://example.test/doc" name="remote doc" />',
      },
      { type: 'text', text: '<resource uri="memory://note/1">remember this</resource>' },
    ]);
  });
});

describe('compressPromptImageParts', () => {
  const trash: string[] = [];

  afterEach(async () => {
    await Promise.all(trash.splice(0).map((dir) => rm(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 })));
  });

  async function tempOriginalsDir(): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), 'acp-originals-'));
    trash.push(dir);
    return dir;
  }

  function imagePart(url: string): ContentPart {
    return { type: 'image_url', imageUrl: { url } };
  }

  it('leaves format judgment to the engine: a format it cannot re-encode passes through', async () => {
    // Which formats are acceptable depends on the provider the agent is
    // bound to, which this edge does not know; the engine's prompt gate
    // decides. So neither a HEIC payload nor a MIME alias is rewritten here.
    const heic = `data:image/heic;base64,${Buffer.alloc(32).toString('base64')}`;
    const alias = `data:IMAGE/PNG;base64,${solidPngBase64(8, 8)}`;
    const out = await compressPromptImageParts([
      { type: 'text', text: 'look' },
      imagePart(heic),
      imagePart(alias),
    ]);
    expect(out).toEqual([{ type: 'text', text: 'look' }, imagePart(heic), imagePart(alias)]);
  });

  it('passes an under-limit image through unchanged and persists nothing', async () => {
    const originalsDir = await tempOriginalsDir();
    const url = `data:image/png;base64,${solidPngBase64(32, 32)}`;
    const out = await compressPromptImageParts([imagePart(url)], {
      originalsDir,
      maxImageEdgePx: 64,
    });
    expect(out).toHaveLength(1);
    expect(out[0]).toEqual(imagePart(url));
    // No compression happened, so no original was persisted.
    expect(await readdir(originalsDir)).toEqual([]);
  });

  it('compresses an over-edge image, prefixes a caption, and persists the original', async () => {
    const originalsDir = await tempOriginalsDir();
    const original = solidPng(128, 128);
    const url = `data:image/png;base64,${original.toString('base64')}`;
    const out = await compressPromptImageParts([imagePart(url)], {
      originalsDir,
      maxImageEdgePx: 64,
    });

    // caption text part immediately precedes the re-encoded image part.
    expect(out).toHaveLength(2);
    const caption = out[0] as { text: string };
    const image = out[1] as { imageUrl: { url: string } };
    expect(caption.text).toContain('Image compressed to fit model limits');
    expect(caption.text).toContain('128x128');
    expect(caption.text).toContain(originalsDir);
    expect(image.imageUrl.url).not.toBe(url);
    expect(image.imageUrl.url.startsWith('data:image/')).toBe(true);

    // The original bytes landed in the session media-originals dir so the
    // model can read fine detail back from the captioned path.
    const files = await readdir(originalsDir);
    expect(files).toHaveLength(1);
    expect(caption.text).toContain(files[0]!);
    expect(await readFile(join(originalsDir, files[0]!))).toEqual(original);
  });
});

describe('displayBlockToAcpContent', () => {
  it('maps a Write file_io display (content only) to an inline diff with null oldText', () => {
    expect(
      displayBlockToAcpContent({
        kind: 'file_io',
        operation: 'write',
        path: '/tmp/notes.md',
        content: '# hello\n',
      }),
    ).toEqual({ type: 'diff', path: '/tmp/notes.md', oldText: null, newText: '# hello\n' });
  });

  it('prefers after over content when both are present', () => {
    expect(
      displayBlockToAcpContent({
        kind: 'file_io',
        operation: 'edit',
        path: '/tmp/a.ts',
        before: 'old',
        after: 'new',
        content: 'ignored',
      }),
    ).toEqual({ type: 'diff', path: '/tmp/a.ts', oldText: 'old', newText: 'new' });
  });

  it('returns null for a Read file_io display (no content to diff)', () => {
    expect(
      displayBlockToAcpContent({ kind: 'file_io', operation: 'read', path: '/tmp/a.ts' }),
    ).toBeNull();
  });
});

describe('structured-kind tool call session updates carry no args preview', () => {
  const editDisplay = {
    kind: 'file_io',
    operation: 'edit',
    path: '/tmp/a.ts',
    before: 'old',
    after: 'new',
  } as const;
  const startedEvent = (over: Record<string, unknown> = {}) => ({
    agentId: 'agent-1',
    turnId: 1,
    toolCallId: 'Edit_1',
    name: 'Edit',
    args: { path: '/tmp/a.ts', old_string: 'old', new_string: 'new' },
    ...over,
  }) as never;

  it('start with a diff display carries only the diff, no args JSON', () => {
    const update = toolCallStartToSessionUpdate('s', startedEvent({ display: editDisplay })).update as any;
    expect(update.content).toEqual([{ type: 'diff', path: '/tmp/a.ts', oldText: 'old', newText: 'new' }]);
  });

  it('start without a display suppresses the args preview for structured kinds', () => {
    const update = toolCallStartToSessionUpdate('s', startedEvent({ name: 'Bash', args: { command: 'ls' } })).update as any;
    expect(update.content).toEqual([]);
  });

  it('start without a display keeps the args preview for unknown (other) tools', () => {
    const update = toolCallStartToSessionUpdate('s', startedEvent({ name: 'Task', args: { prompt: 'patrol' } })).update as any;
    expect(update.content).toEqual([{ type: 'content', content: { type: 'text', text: '{"prompt":"patrol"}' } }]);
  });

  it('started-upgrade after a lazy create also drops the args preview when a diff exists', () => {
    const update = toolCallStartedUpgradeToSessionUpdate('s', startedEvent({ display: editDisplay })).update as any;
    expect(update.content).toEqual([{ type: 'diff', path: '/tmp/a.ts', oldText: 'old', newText: 'new' }]);
  });

  it('started-upgrade without a display suppresses the args preview for structured kinds', () => {
    const update = toolCallStartedUpgradeToSessionUpdate('s', startedEvent({ name: 'Read', args: { path: '/tmp/a.ts' } })).update as any;
    expect(update.content).toEqual([]);
  });

  it('lazy create for an edit-kind tool starts with empty content', () => {
    const update = toolCallLazyCreateToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 'Edit_1', name: 'Edit', argumentsPart: '{"path"',
    } as never).update as any;
    expect(update.content).toEqual([]);
  });

  it('lazy create for a shell tool starts with empty content', () => {
    const update = toolCallLazyCreateToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 'Bash_1', name: 'Bash', argumentsPart: '{"command"',
    } as never).update as any;
    expect(update.content).toEqual([]);
  });

  it('lazy create maps Glob/Grep to the search kind', () => {
    const update = toolCallLazyCreateToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 'Grep_1', name: 'Grep', argumentsPart: '{"pattern"',
    } as never).update as any;
    expect(update.kind).toBe('search');
    expect(update.content).toEqual([]);
  });

  it('lazy create for other tools keeps the streaming fragment', () => {
    const update = toolCallLazyCreateToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 'Task_1', name: 'Task', argumentsPart: '{"prompt"',
    } as never).update as any;
    expect(update.content).toEqual([{ type: 'content', content: { type: 'text', text: '{"prompt"' } }]);
  });

  it('deltas for structured-kind tools stay empty once the kind is seen, even when the name arrives late', () => {
    const acc = { args: '' };
    const first = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't1', argumentsPart: '{"path"',
    } as never, acc).update as any;
    expect(first.content).toHaveLength(1);
    const second = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't1', name: 'Edit', argumentsPart: ',"old_string":"x"',
    } as never, acc).update as any;
    expect(second.content).toEqual([]);
    const third = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't1', argumentsPart: ',"new_string":"y"}',
    } as never, acc).update as any;
    expect(third.content).toEqual([]);
    expect(acc.args).toBe('{"path","old_string":"x","new_string":"y"}');
  });

  it('deltas for shell tools stay empty', () => {
    const acc = { args: '' };
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't3', name: 'Bash', argumentsPart: '{"command":"ls"',
    } as never, acc).update as any;
    expect(update.content).toEqual([]);
  });

  it('deltas for other tools keep the cumulative preview', () => {
    const acc = { args: '' };
    toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't2', name: 'Task', argumentsPart: '{"prompt":"fix',
    } as never, acc);
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't2', argumentsPart: ' it"}',
    } as never, acc).update as any;
    expect(update.content).toEqual([{ type: 'content', content: { type: 'text', text: '{"prompt":"fix it"}' } }]);
  });
});

describe('toolResultToSessionUpdate with a diff display', () => {
  const editDisplay = {
    kind: 'file_io',
    operation: 'edit',
    path: '/tmp/a.ts',
    before: 'old',
    after: 'new',
  } as const;
  const resultEvent = (over: Record<string, unknown> = {}) => ({
    agentId: 'agent-1',
    turnId: 1,
    toolCallId: 'Edit_1',
    output: 'File updated',
    ...over,
  }) as never;
  const diffBlock = { type: 'diff', path: '/tmp/a.ts', oldText: 'old', newText: 'new' };

  it('a successful structured edit carries only the diff; rawOutput preserves the result', () => {
    const update = toolResultToSessionUpdate('s', resultEvent(), undefined, editDisplay).update as any;
    expect(update.status).toBe('completed');
    expect(update.content).toEqual([diffBlock]);
    expect(update.rawOutput).toBe('File updated');
  });

  it('a failed edit keeps the error text after the diff', () => {
    const update = toolResultToSessionUpdate('s', resultEvent({ isError: true, output: 'old_string not found' }), undefined, editDisplay).update as any;
    expect(update.status).toBe('failed');
    expect(update.content).toEqual([diffBlock, { type: 'content', content: { type: 'text', text: 'old_string not found' } }]);
  });

  it('a successful call without a display carries the result text', () => {
    const update = toolResultToSessionUpdate('s', resultEvent({ output: 'done' })).update as any;
    expect(update.content).toEqual([{ type: 'content', content: { type: 'text', text: 'done' } }]);
  });
});

describe('toolCallLocations by tool kind', () => {
  const startedEvent = (over: Record<string, unknown> = {}) => ({
    agentId: 'agent-1',
    turnId: 1,
    toolCallId: 'c1',
    name: 'Edit',
    args: { path: '/tmp/a.ts', old_string: 'old', new_string: 'new' },
    ...over,
  }) as never;

  it('drops locations for search tools (the search root is not a result file)', () => {
    const grep = toolCallStartToSessionUpdate('s', startedEvent({ name: 'Grep', args: { pattern: 'x', path: '/tmp' } })).update as any;
    expect(grep.locations).toBeUndefined();
    const glob = toolCallStartToSessionUpdate('s', startedEvent({ name: 'Glob', args: { pattern: '*.ts', path: '/tmp' } })).update as any;
    expect(glob.locations).toBeUndefined();
  });

  it('keeps locations for file tools (a real file link)', () => {
    const read = toolCallStartToSessionUpdate('s', startedEvent({ name: 'Read', args: { path: '/tmp/a.ts' } })).update as any;
    expect(read.locations).toEqual([{ path: '/tmp/a.ts' }]);
    const edit = toolCallStartToSessionUpdate('s', startedEvent()).update as any;
    expect(edit.locations).toEqual([{ path: '/tmp/a.ts' }]);
  });
});

describe('toolCallDeltaToSessionUpdate rawInput forwarding', () => {
  it('forwards rawInput as soon as the streaming args JSON parses, preview text still suppressed', () => {
    const acc = { args: '' };
    toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't1', name: 'Bash', argumentsPart: '{"command":"ec',
    } as never, acc);
    const second = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't1', name: 'Bash', argumentsPart: 'ho hi"}',
    } as never, acc).update as any;
    expect(second.rawInput).toEqual({ command: 'echo hi' });
    expect(second.content).toEqual([]);
  });

  it('omits rawInput while the args are incomplete JSON', () => {
    const acc = { args: '' };
    const first = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't2', name: 'Bash', argumentsPart: '{"command":"ec',
    } as never, acc).update as any;
    expect(first.rawInput).toBeUndefined();
  });
});

describe('toolCallDeltaToSessionUpdate live edit diffs', () => {
  it('emits an old→new diff block for Edit once the args parse', () => {
    const acc = { args: '' };
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't3', name: 'Edit',
      argumentsPart: '{"path":"/tmp/a.ts","old_string":"before","new_string":"after"}',
    } as never, acc).update as any;
    expect(update.content).toEqual([{ type: 'diff', path: '/tmp/a.ts', oldText: 'before', newText: 'after' }]);
    expect(update.rawInput).toEqual({ path: '/tmp/a.ts', old_string: 'before', new_string: 'after' });
  });

  it('emits a null-oldText diff for Write content', () => {
    const acc = { args: '' };
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't4', name: 'Write',
      argumentsPart: '{"path":"/tmp/n.md","content":"# hi\\n"}',
    } as never, acc).update as any;
    expect(update.content).toEqual([{ type: 'diff', path: '/tmp/n.md', oldText: null, newText: '# hi\n' }]);
  });
});

describe('toolCallDeltaToSessionUpdate title preview', () => {
  it('previews the growing args tail in the title while the JSON is incomplete', () => {
    const acc = { args: '' };
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't5', name: 'Bash', argumentsPart: '{"command":"npm test -- --wa',
    } as never, acc).update as any;
    expect(update.title).toBe('Generating args… {"command":"npm test -- --wa');
    expect(update.content).toEqual([]);
    expect(update.rawInput).toBeUndefined();
  });

  it('trims the preview to the last 40 chars and collapses whitespace', () => {
    const acc = { args: '' };
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't6', name: 'Task',
      argumentsPart: '{"prompt":"' + 'x'.repeat(60) + '\n\n next',
    } as never, acc).update as any;
    expect(update.title.startsWith('Generating args… …')).toBe(true);
    expect(update.title.length).toBeLessThanOrEqual('Generating args… …'.length + 40);
    expect(update.title).not.toContain('\n');
    expect(update.content[0].content.text).toContain('x'.repeat(60));
  });

  it('drops the preview title once the args parse', () => {
    const acc = { args: '' };
    toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't7', name: 'Bash', argumentsPart: '{"command":"ec',
    } as never, acc);
    const update = toolCallDeltaToSessionUpdate('s', {
      agentId: 'a', turnId: 1, toolCallId: 't7', name: 'Bash', argumentsPart: 'ho hi"}',
    } as never, acc).update as any;
    expect(update.title).toBeUndefined();
    expect(update.rawInput).toEqual({ command: 'echo hi' });
  });
});
