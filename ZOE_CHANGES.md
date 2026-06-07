# Zoe's Changes — Cross-Year ESG Agent

Branch: `feature/cross-year-agent`. Scope: give the existing ESG agent cross-year
(2022 / 2023 / 2024) capability, add a dedicated comparison page, and tidy the app
for the presentation. **No original `outputs/**` CSV/NPZ was modified — read-only.**

## What changed

### A. Cross-year Q&A retrieval
- `agents/corpus.py` (new): loads all three years' `chunks_processed.csv`, tags every
  chunk with its `year`, and pools them with a globally-unique `chunk_id` (`<year>#<id>`).
  Cached so the CSVs are read at most once per process.
- `agents/retrieve.py`: TF-IDF cosine retrieval now runs over year-tagged records and
  returns each chunk with its `year`. `qa_scope="cross_year"` pools all three years
  (default); otherwise it stays single-year.
- `agents/llm.py`: the Q&A prompt tags each chunk as `[Year YYYY | Chunk N]`, and when
  evidence spans several years it is told to compare them explicitly. Grounding is kept:
  answer only from retrieved chunks. The no-key fallback also shows the year tags.
- `agents/graph.py`: `answer_question(state, question, scope=...)` threads the scope;
  default is cross-year.
- `agents/state.py`: added `qa_scope` / `qa_years`.

> Design note: the cross-year pooling lives in `retrieval_agent`, so the single-year
> dashboard pipeline (keywords / classification / summary) is untouched and stays fast.
> The three CSVs are only loaded when a cross-year question is actually asked.

### B. Cross-year analysis as a reusable tool
- `cross_year_analysis.py`: refactored from a one-off script into reusable functions
  (`get_cross_year_metrics(sdg)`, `get_cross_year_payload()`, `load_cross_year_payload()`).
  Running `python cross_year_analysis.py` regenerates BOTH derived files in one command:
  - `outputs/cross_year_analysis.md` (human-readable report)
  - `outputs/cross_year_analysis.json` (snapshot read by the app + tools — fast, demo-stable)
- `src/agent_tools.py`: added `get_cross_year_trend(sdg)`, following the repo's existing
  tool convention (returns the standard `tool / group_name / rows / summary` dict). It reads
  the cached snapshot — it does not recompute.
- `agents/graph.py` + `agents/llm.py`: `summarize_cross_year(...)` plus a
  `mode="cross_year:<SDG>"` summary mode. The prompt is strictly limited to describing the
  supplied numbers (coverage shares, persistent/new/dropped keywords, centroid distances);
  it must not speculate about motives. Deterministic fallback when there is no API key.

### C. Cross-Year Compare page (new top-level page)
- `app_pages/cross_year_panel.py` (new): pick one of the 9 SDGs, then see
  - **Change in quantity** — coverage share per year (bar chart) + Δ share.
  - **Change in quality | semantic centroid shift** — how far the topic's average
    embedding moved year to year (Total shift, 2022→2023, 2023→2024, 2022→2024 direct).
  - **Change in quality | Top keywords side by side** — three years aligned, colour-coded
    Persistent / New / Dropped.
  - **Cross-year narrative** — grounded, no-key-safe.
  Reads only the cached JSON snapshot (`st.cache_data`); the front end never recomputes.
- `app.py`: registered the page.

### Housekeeping
- `app.py`: sidebar **Page** list trimmed to the two demo pages
  (**Final Project Agents**, **Cross-Year Compare**). Other pages' code/imports were kept,
  so this is fully reversible — just add the names back.
- `app_pages/final_project_agents.py`: Q&A gained a **Retrieval scope** selector defaulting
  to *All years (2022-2024)*; source chunks now show their year; hero text simplified.
- All web-facing strings are English.
- `main.py`: fixed a stray `i` on line 1 (`i # main.py` → `# main.py`) that made
  `python main.py` raise `NameError`. Pre-existing, unrelated to this feature, fixed in passing.
- `app_pages/final_project_agents.py`: the cached agent state is now re-synced with the
  current sidebar key/model each rerun, so a freshly typed API key takes effect for Q&A
  without re-running the whole analysis.

## Round 2 — UI cleanup + LLM hardening

### LLM path is now testable
- The OpenAI client libraries (`openai`, `langchain-openai`, `langgraph`) are listed in
  `requirements.txt`. Install them to enable real LLM output:
  ```bash
  pip install openai langchain-openai langgraph
  ```
  Then enter the key in the sidebar (or set `OPENAI_API_KEY`). With no key / no libraries,
  everything still runs in deterministic fallback mode — the demo-day safety net.
- **Error transparency** (`agents/llm.py`): the LLM helpers used to swallow every error and
  always print "Without an API key". They now surface the real reason — e.g. a `401`
  (invalid key) or `429` (quota exceeded) is shown in the answer, so you can tell a quota
  problem from a code problem. (A `429 RateLimitError` means the OpenAI account is out of
  quota — add billing; it is not a code issue.)
- **Q&A key sync** (`final_project_agents.py`): the cached agent state is re-synced with the
  current sidebar key/model on every rerun, so a freshly typed key works for Q&A without
  re-running the whole analysis. The sidebar shows whether LLM or fallback mode is active.

### Cross-year narrative quality
- `agents/llm.summarize_cross_year`: the prompt was rewritten to read like an analyst, not a
  CSV dump — it is fed clean, readable fields (not a raw JSON blob) and asked to lead with the
  most notable change and tie the keyword turnover to the semantic shift, in two short
  paragraphs. Still grounded: describe WHAT changed, not WHY; no invented facts or motives.
- `agents/llm.fallback_summary`: dropped the meta "the dashboard combines…" sentence so the
  no-key summary reads about the report, not about the system.

### Single-year vs cross-year split (clearer page roles)
- **Final Project Agents = single year**, **Cross-Year Compare = three years.** Removed the
  cross-year sample questions and the Retrieval-scope selector from the Q&A; it now answers
  only the sidebar's selected year, with a one-line pointer to the Cross-Year Compare page.
  (Cross-year retrieval capability is retained in `corpus.py` / `retrieve.py`, just not
  surfaced here — ready for an agent to call later.)

### Final Project Agents page redesign (AI-first, light touch)
- The AI summary is promoted out of a tab into a prominent bordered card above the fold,
  labelled with which agent wrote it, the style, and whether it used the LLM or the fallback.
- "Ask the report" is now its own prominent section (no longer buried in a tab).
- The deterministic detail (Keywords table + ESG evidence) is collapsed into a single
  `Evidence & analysis details` expander, so the AI stays the focus but the proof is one click away.
- Removed the redundant (and visually broken, double-stacked) keyword bar chart; the word
  cloud already is the visual version of the keyword table.
- Hid the always-empty `frequency` column and added captions clarifying that the keyword
  `group` (report section) is a different scheme from the E/S/G classification.
- Simplified the hero banner.

## Suggested next step (for Kiwi)
This will look much cooler once **Kiwi's agent workflow is wired in on top**: instead of the
deterministic pipeline calling the cross-year tool directly, let the agent *decide* to call
`get_cross_year_trend` / cross-year retrieval as tools in response to a user question. The
plumbing is already in place — `src/agent_tools.get_cross_year_trend` follows the same tool
contract the Week 13 AG2 agent already uses, so it can be registered as one more tool and the
agent can choose it for "how did topic X change over the years" questions. That turns the
current buttons into an agent-driven flow, which is a stronger story for the demo.
