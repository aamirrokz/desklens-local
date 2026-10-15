from __future__ import annotations

import io
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

st.set_page_config(page_title="DeskLens Local", page_icon="🛠️", layout="wide")

SAMPLE_TICKETS = pd.DataFrame(
    [
        {
            "ticket_id": "INC-1041",
            "short_description": "Outlook keeps asking for password",
            "description": "User is prompted to sign in repeatedly after changing their corporate password.",
            "resolution_notes": "Confirmed password change had not refreshed in the desktop session. Signed out of Office apps, closed Outlook, refreshed credentials through the approved sign-in flow, then reopened Outlook. User confirmed mail was syncing.",
            "state": "Closed",
            "category": "Email",
        },
        {
            "ticket_id": "INC-1042",
            "short_description": "VPN connects but internal sites do not open",
            "description": "VPN client reports connected, but intranet pages time out. Public websites work.",
            "resolution_notes": "Verified the user was on the correct VPN profile. Reconnected using the approved profile and refreshed DNS using the standard support procedure. Intranet access restored.",
            "state": "Resolved",
            "category": "Network",
        },
        {
            "ticket_id": "INC-1043",
            "short_description": "Teams microphone not detected",
            "description": "Built-in microphone is missing from Teams device settings after a laptop restart.",
            "resolution_notes": "Checked operating-system privacy permissions and selected the built-in microphone in Teams settings. Test call confirmed audio input worked.",
            "state": "Closed",
            "category": "Collaboration",
        },
        {
            "ticket_id": "INC-1044",
            "short_description": "Laptop running slowly after startup",
            "description": "Device is slow for several minutes after login; CPU usage is elevated.",
            "resolution_notes": "Reviewed Task Manager and startup applications, applied approved pending updates, restarted the device, and confirmed normal performance. Escalation criteria were documented if the issue recurs.",
            "state": "Closed",
            "category": "Endpoint",
        },
        {
            "ticket_id": "INC-1045",
            "short_description": "OneDrive files stuck syncing",
            "description": "Sync icon remains pending for several files. User can browse the web normally.",
            "resolution_notes": "Checked storage and file-name restrictions, then paused and resumed sync using the client controls. Sync completed for the affected files.",
            "state": "Resolved",
            "category": "Collaboration",
        },
        {
            "ticket_id": "INC-1046",
            "short_description": "Printer shows offline",
            "description": "Office printer is unavailable from one laptop, but coworkers can print.",
            "resolution_notes": "Removed and re-added the approved printer queue, verified the selected queue, and printed a test page successfully.",
            "state": "Closed",
            "category": "Peripherals",
        },
        {
            "ticket_id": "INC-1047",
            "short_description": "Outlook search returns no recent messages",
            "description": "Mail is arriving, but searches do not return messages received today.",
            "resolution_notes": "Verified search scope and indexing status. Rebuilt the local search index using the organization's standard process and re-tested with the user.",
            "state": "Closed",
            "category": "Email",
        },
        {
            "ticket_id": "INC-1048",
            "short_description": "MFA prompt repeats when signing in",
            "description": "User completes multi-factor authentication but is returned to the sign-in screen.",
            "resolution_notes": "Checked the reported time and application, then followed the approved identity support runbook to clear the affected session and sign in again. Referred to identity support when the issue recurred.",
            "state": "In Progress",
            "category": "Identity",
        },
    ]
)


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        c = column.lower().strip().replace("_", " ")
        if any(candidate in c for candidate in candidates):
            return column
    return None


def normalize_tickets(
    df: pd.DataFrame,
    title_col: str,
    description_col: str | None,
    resolution_col: str | None,
    state_col: str | None,
    id_col: str | None,
) -> pd.DataFrame:
    normalized = pd.DataFrame()
    normalized["ticket_id"] = df[id_col].fillna("").astype(str) if id_col else [f"ROW-{i + 1}" for i in range(len(df))]
    normalized["short_description"] = df[title_col].fillna("").astype(str)
    normalized["description"] = df[description_col].fillna("").astype(str) if description_col else ""
    normalized["resolution_notes"] = df[resolution_col].fillna("").astype(str) if resolution_col else ""
    normalized["state"] = df[state_col].fillna("").astype(str) if state_col else ""
    normalized = normalized[normalized["short_description"].str.strip().ne("")].reset_index(drop=True)
    normalized["search_text"] = (normalized["short_description"] + " " + normalized["description"]).str.strip()
    resolved_state = normalized["state"].str.lower().str.contains(r"resolved|closed|completed|fixed", regex=True, na=False)
    has_resolution = normalized["resolution_notes"].str.strip().ne("")
    normalized["is_resolved"] = resolved_state | has_resolution
    return normalized


def find_similar_tickets(tickets: pd.DataFrame, title: str, description: str, limit: int = 5) -> pd.DataFrame:
    query = f"{title} {description}".strip()
    candidates = tickets[
        tickets["is_resolved"] & tickets["resolution_notes"].astype(str).str.strip().ne("")
    ].copy()
    if candidates.empty or not query:
        return candidates.head(0).assign(similarity=pd.Series(dtype=float))

    corpus = candidates["search_text"].fillna("").astype(str).tolist() + [query]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return candidates.head(0).assign(similarity=pd.Series(dtype=float))

    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    candidates["similarity"] = scores
    return candidates.sort_values("similarity", ascending=False).head(limit)


def generate_local_draft(model: str, title: str, description: str, similar: pd.DataFrame) -> str:
    evidence: list[dict[str, str]] = []
    for _, row in similar.iterrows():
        evidence.append(
            {
                "ticket_id": str(row.get("ticket_id", "")),
                "title": str(row.get("short_description", "")),
                "resolution": str(row.get("resolution_notes", "")),
            }
        )

    system_prompt = (
        "You are DeskLens Local, a cautious IT service-desk triage assistant. "
        "Draft suggestions for a human analyst; never claim that a step has already been completed. "
        "Use only the provided ticket details and historical resolutions as evidence. "
        "Do not invent company policies, credentials, configuration values, or access permissions. "
        "If the evidence is weak or incomplete, say what must be verified or ask a clarifying question. "
        "Prefer reversible, low-risk checks; flag steps that require an administrator or approved runbook. "
        "Return these headings: Summary, Checks to try, Suggested user reply, Escalate when."
    )
    user_prompt = (
        f"New ticket title: {title}\n"
        f"New ticket description: {description}\n\n"
        f"Similar resolved tickets (historical evidence; may not apply exactly): {evidence}\n\n"
        "Create a concise draft for the support analyst. Cite any historical ticket IDs you rely on. "
        "If no relevant historical evidence is present, make that limitation explicit."
    )

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return payload.get("message", {}).get("content", "The local model returned an empty response.")
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "Could not connect to Ollama on this computer. Start Ollama and run `ollama pull qwen2.5:7b`."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The local model took too long to respond. Try a smaller model or shorter ticket text.") from exc
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.text[:400]
        except Exception:
            pass
        raise RuntimeError(f"Ollama returned an error: {detail or exc}") from exc


st.title("DeskLens Local")
st.caption("Private-by-default ticket triage for small IT teams and managed service providers")
st.write(
    "Search your resolved incidents for similar cases, then use a locally running open-weight model "
    "to draft cautious troubleshooting steps and a user reply. A human analyst reviews every suggestion."
)

with st.sidebar:
    st.header("Local setup")
    model_name = st.text_input("Ollama model", value=DEFAULT_MODEL, help="Example: qwen2.5:7b or another model installed in Ollama.")
    st.caption("AI inference goes to Ollama at 127.0.0.1 on this computer—not to a hosted AI API.")
    uploaded = st.file_uploader("Optional: upload a ticket CSV", type=["csv"])
    st.caption("Use synthetic or organization-approved data. Remove passwords, tokens, personal data, and secrets before importing.")

source_name = "built-in synthetic demo data"
raw_df = SAMPLE_TICKETS.copy()
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
        st.stop()

columns = list(raw_df.columns)
if not columns:
    st.error("The selected CSV has no columns.")
    st.stop()

with st.sidebar:
    st.header("Map CSV columns")
    guessed_title = find_column(columns, ["short description", "short_description", "title", "subject", "summary", "incident"])
    title_choice = st.selectbox("Ticket title (required)", columns, index=columns.index(guessed_title) if guessed_title in columns else 0)

    def select_optional(label: str, guesses: list[str]) -> str | None:
        guessed = find_column(columns, guesses)
        options = ["— None —"] + columns
        index = options.index(guessed) if guessed in columns else 0
        chosen = st.selectbox(label, options, index=index)
        return None if chosen == "— None —" else chosen

    description_choice = select_optional("Description", ["description", "details", "issue", "symptoms"])
    resolution_choice = select_optional("Resolution notes", ["resolution notes", "resolution_notes", "resolution", "fix notes", "close notes"])
    state_choice = select_optional("State / status", ["state", "status", "ticket status"])
    id_choice = select_optional("Ticket ID", ["ticket id", "ticket_id", "number", "incident number", "id"])

try:
    tickets = normalize_tickets(raw_df, title_choice, description_choice, resolution_choice, state_choice, id_choice)
except Exception as exc:
    st.error(f"Could not prepare the ticket data: {exc}")
    st.stop()

if tickets.empty:
    st.error("No tickets with a non-empty title were found. Check the column mapping and CSV contents.")
    st.stop()

resolved_count = int(tickets["is_resolved"].sum())
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Tickets loaded", len(tickets))
metric2.metric("With resolution history", resolved_count)
metric3.metric("Data source", "CSV upload" if uploaded is not None else "Demo")
st.caption(f"Current source: {source_name}. Ticket data is processed in this Streamlit app; no database or cloud API is configured.")

st.divider()
st.subheader("Triage a ticket")
with st.form("triage_form"):
    ticket_title = st.text_input("Short description", placeholder="e.g., VPN connects but intranet is unavailable")
    ticket_description = st.text_area("Issue details", height=130, placeholder="What is happening, what changed, and any error message? Remove sensitive details.")
    submitted = st.form_submit_button("Find similar cases", type="primary")

if submitted:
    if not ticket_title.strip():
        st.warning("Enter a short description to search for similar cases.")
    else:
        matches = find_similar_tickets(tickets, ticket_title, ticket_description, limit=5)
        st.session_state["triage_input"] = {"title": ticket_title.strip(), "description": ticket_description.strip()}
        st.session_state["triage_matches"] = matches

if "triage_input" in st.session_state:
    triage_input = st.session_state["triage_input"]
    matches = st.session_state.get("triage_matches", pd.DataFrame())
    st.subheader("Similar resolved cases")
    if matches.empty:
        st.info("No resolved cases with resolution notes matched. Add useful resolution history or refine the ticket wording.")
    else:
        visible = matches[["ticket_id", "short_description", "resolution_notes", "similarity"]].copy()
        visible["similarity"] = (visible["similarity"] * 100).round(1).astype(str) + "%"
        st.dataframe(visible, use_container_width=True, hide_index=True)

    st.subheader("Draft a local AI triage suggestion")
    st.write("The model receives the new ticket and only the retrieved historical cases shown above.")
    if st.button("Generate draft with local model"):
        with st.spinner("Generating on this computer…"):
            try:
                draft = generate_local_draft(model_name.strip() or DEFAULT_MODEL, triage_input["title"], triage_input["description"], matches)
                st.session_state["latest_draft"] = draft
            except RuntimeError as exc:
                st.error(str(exc))

    if st.session_state.get("latest_draft"):
        st.markdown(st.session_state["latest_draft"])
        st.download_button(
            "Download draft as Markdown",
            data=st.session_state["latest_draft"],
            file_name="desklens-triage-draft.md",
            mime="text/markdown",
        )

with st.expander("What this prototype does—and does not do"):
    st.markdown(
        """
        **Does:** locally rank similar resolved tickets, pass a small set of matches to a local model, and produce a draft for analyst review.

        **Does not:** connect to ServiceNow, close or modify incidents, validate a fix, or guarantee that a historical resolution fits a new issue.

        **Privacy note:** inference is configured for the local Ollama endpoint. The app does not intentionally send ticket contents to an external service, but users should still follow their employer's data-handling rules and review their local environment before using real tickets.
        """
    )
