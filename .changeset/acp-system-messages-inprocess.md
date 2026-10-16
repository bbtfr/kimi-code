---
"@moonshot-ai/kimi-code": patch
---

Fold the system-messages ACP proxy into the ACP server: legacy `<paseo-system>` schedule prompts now project to display envelopes in-process (prompt mirror, echo rewrite, fragment buffering, cold-load replay), and completed edit results drop text blocks that exactly duplicate `rawOutput`, so Paseo no longer needs a proxy process in the launch command.
