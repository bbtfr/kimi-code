import {
  createContextTranscriptReducer,
  ensureMainAgent,
  getLiveSessionById,
  IAgentBlobService,
  IAgentLifecycleService,
  IWireService,
  type Scope,
  type ContentPart,
} from '@moonshot-ai/agent-core-v2';
import type { ToolResultEvent } from '@moonshot-ai/agent-core-v2/events';
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
  const toolResults = new Map<string, ToolResultEvent>();
  for await (const record of wire.read()) {
    reducer.add(record);
    if (record.type !== 'context.append_loop_event') continue;
    const event = record['event'] as Record<string, unknown> | undefined;
    if (event?.['type'] === 'tool.result' && typeof event['toolCallId'] === 'string') {
      const call = toolCalls.get(event['toolCallId']);
      const result = event['result'] as { output: unknown; isError?: boolean };
      if (call !== undefined) toolResults.set(event['toolCallId'], {
        type: 'tool.result', turnId: call.turnId, toolCallId: call.toolCallId,
        output: result.output, isError: result.isError,
      });
    }
    if (event?.['type'] !== 'tool.call' || typeof event['toolCallId'] !== 'string' || typeof event['name'] !== 'string') continue;
    const turnId = Number(event['turnId']);
    if (!Number.isFinite(turnId)) continue;
    toolCalls.set(event['toolCallId'], {
      type: 'tool.call.started',
      turnId,
      toolCallId: event['toolCallId'],
      name: event['name'],
      args: event['args'],
      description: typeof event['description'] === 'string' ? event['description'] : undefined,
      display: event['display'] as ToolInputDisplay | undefined,
    });
  }
  const blobs = agent.accessor.get(IAgentBlobService);
  const messages = await Promise.all(reducer.result().entries.map(async (message) => ({
    ...message,
    content: [...await blobs.loadParts(message.content)],
  })));
  for (const [id, result] of toolResults) {
    if (Array.isArray(result.output)) {
      toolResults.set(id, { ...result, output: await blobs.loadParts(result.output as ContentPart[]) });
    }
  }
  return { messages, toolCalls, toolResults };
}
