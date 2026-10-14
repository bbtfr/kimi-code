import {
  createContextTranscriptReducer,
  ensureMainAgent,
  getLiveSessionById,
  IAgentBlobService,
  IAgentLifecycleService,
  IWireService,
  type Scope,
} from '@moonshot-ai/agent-core-v2';
import type { ToolCallStartedEvent } from '@moonshot-ai/agent-core-v2/agent/toolExecutor/toolExecutorEvents';
import type { ToolInputDisplay } from '@moonshot-ai/agent-core-v2/tool/toolInputDisplay';
import type { AcpHistory } from './replay';

/** Read the active journal branch, retaining persisted display semantics across compaction. */
export async function readAcpHistory(core: Scope, sessionId: string): Promise<AcpHistory> {
  const session = getLiveSessionById(core.accessor, sessionId);
  if (session === undefined) throw new Error(`Session ${sessionId} is not live`);
  const context = await ensureMainAgent(session);
  const agent = session.accessor.get(IAgentLifecycleService).handleOf(context.agentId)!;
  const wire = agent.accessor.get(IWireService);
  await wire.flush();
  const reducer = createContextTranscriptReducer();
  const toolCalls = new Map<string, ToolCallStartedEvent>();
  for await (const record of wire.read()) {
    reducer.add(record);
    if (record.type !== 'context.append_loop_event') continue;
    const event = record['event'] as Record<string, unknown> | undefined;
    if (event?.['type'] !== 'tool.call' || typeof event['toolCallId'] !== 'string' || typeof event['name'] !== 'string') continue;
    const turnId = Number(event['turnId']);
    if (!Number.isFinite(turnId)) continue;
    toolCalls.set(event['toolCallId'], {
      type: 'tool.call.started',
      turnId,
      toolCallId: event['toolCallId'],
      name: event['name'],
      args: event['args'],
      display: event['display'] as ToolInputDisplay | undefined,
    });
  }
  const blobs = agent.accessor.get(IAgentBlobService);
  const messages = await Promise.all(reducer.result().entries.map(async (message) => ({
    ...message,
    content: [...await blobs.loadParts(message.content)],
  })));
  return { messages, toolCalls };
}
