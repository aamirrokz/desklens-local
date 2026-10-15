# DeskLens Local

**A local-first IT support triage assistant for small IT teams and managed service providers.**

DeskLens Local searches a ticket export for similar resolved incidents and uses a locally running open-weight language model to draft troubleshooting checks, an end-user reply, and escalation criteria. It is an analyst aid, not an autonomous ticket closer.

## Why someone might pay

Small IT teams often repeat the same investigation across incidents but cannot justify an enterprise knowledge-management platform. A lightweight local tool can help surface what worked before without sending ticket histories to a hosted AI service.

Possible commercial model to validate: a one-time paid setup for importing and mapping each client's CSV export, followed by a per-device or per-analyst license for updates and support. Interview 5–10 IT providers before setting a price; no willingness-to-pay research has been conducted for this prototype.

## What is included

- `app.py`: Streamlit interface, local similarity search, and Ollama integration.
- `sample_tickets.csv`: synthetic tickets for a safe demo.
- `requirements.txt`: Python dependencies.
- `submission_post.md`: a challenge-entry draft.

## Run locally

Requirements: Python 3.10+ and [Ollama](https://ollama.com/).

1. Install Ollama and download a model once:

   ```bash
   ollama pull qwen2.5:7b
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. Try the sample tickets, enter a new ticket such as "VPN connects but intranet pages time out", and generate a draft. Then upload a CSV and map its columns in the sidebar.

After software and model downloads have completed, the app and inference can run without an internet connection, provided Python packages and the Ollama model are already installed locally. Model performance depends on the computer's available memory and hardware.

## Privacy and safety boundaries

- The model call targets `127.0.0.1` only; no hosted AI API is configured.
- The app does not persist uploaded tickets to a database.
- Network requests to local Ollama are only for inference. Installing packages or downloading models needs internet access first.
- Use synthetic or organization-approved, appropriately sanitized ticket data. Do not upload credentials, access tokens, confidential records, or personal data without authorization.
- Review AI drafts before use. Validate commands against approved runbooks. Do not let the prototype close tickets or make production changes automatically.
- This is a prototype, not a security-audited enterprise application.

## Open innovation choices

- **Open-weight model:** can be swapped for another Ollama-compatible model to compare quality, latency, language support, and hardware needs.
- **Local inference:** ticket text stays on the user's machine rather than being sent to a third-party inference API.
- **No per-token API bill:** inference uses local compute, though hardware, electricity, support, and setup still cost money.
- **Adaptable workflow:** a team can modify the prompts, ticket fields, retrieval logic, and escalation policy to suit its own runbooks.

Check and comply with the exact model license for the model and quantization you deploy, especially before commercial distribution. Qwen2.5 model variants do not all share the same license; the 7B model is listed under Apache 2.0 in the Ollama library as of the research used for this prototype.

## Current limitations / next steps

- Similarity search is TF-IDF lexical matching, not semantic embeddings; acronyms and differently worded issues may be missed.
- CSV column mapping is manual and each vendor's export will need testing.
- No ServiceNow integration, role-based access, audit log, encryption-at-rest layer, evaluation benchmark, or multi-user deployment is included.
- Before commercialization, test on sanitized historical data with analysts, track citation/evidence quality and time saved, and add proper access controls and an organization-specific security review.
