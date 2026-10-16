---
"@moonshot-ai/kimi-code": patch
---

Emit ACP system notices (compaction, LLM request retries, engine warnings/errors, and turn failures) as `<notification source="kimi">` envelopes instead of bare assistant text, so clients that understand the convention render them as notifications rather than model output.
