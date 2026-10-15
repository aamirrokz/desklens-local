## DeskLens Local: a private AI copilot for small IT support teams

Small IT teams often investigate the same Outlook, VPN, Teams, and laptop issues over and over. The fix may already exist in a resolved ticket, but finding it takes time.

I built **DeskLens Local**, a prototype that imports a support-ticket CSV, finds similar resolved incidents, and uses a locally running open-weight model to draft troubleshooting checks, a user reply, and escalation criteria. It keeps a human analyst in control: it suggests; it does not close tickets or run fixes.

**Why open AI matters here:** the ticket history can include internal systems, user details, and operational context. DeskLens sends the prompt to a local Ollama endpoint instead of a hosted AI API, and it has no per-token inference bill. Once the model and dependencies are installed, the workflow can run offline. The model and prompt can be swapped or adapted to a team's runbooks without being locked to one provider's API or pricing.

**Who might pay:** small managed service providers and internal IT teams that do not need—or cannot justify—a large service-management platform. The first offer could be a paid setup that maps their ticket export and tests it against sanitized resolved incidents, with ongoing support as an add-on. That pricing hypothesis still needs customer interviews and validation.

**What is open and changeable:** the model, retrieval logic, prompts, and workflow. The prototype currently uses TF-IDF to retrieve similar resolved incidents and Qwen2.5 through Ollama for the draft. It uses synthetic sample tickets for the demo; no real customer data or measured time-savings are being claimed.

The next step is to test it with a few small IT teams using approved, sanitized ticket exports, measure whether analysts resolve repeat issues faster, and improve retrieval before adding ticketing-system integration.

**Repo/demo:** [add your repository or demo link here]
