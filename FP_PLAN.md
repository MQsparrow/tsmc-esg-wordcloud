# Final Project Plan: TSMC ESG Text Mining Agents

## 1. Project Goal

Build a LangGraph-based multi-agent system that analyzes TSMC ESG-related text and produces an interactive report.

The system should help users:

- Upload or select TSMC ESG / sustainability report text.
- Extract important keywords and ESG-related terms.
- Classify content into ESG categories.
- Generate summaries and insights.
- Display word clouds, charts, and an AI-generated final explanation in Streamlit.

Recommended stack:

- UI: Streamlit
- Agent workflow: LangGraph
- LLM provider: OpenAI API first, with optional support for local/Ollama models later
- Text processing: pandas, spaCy or jieba, scikit-learn
- Visualization: matplotlib, wordcloud, plotly
- Optional vector search: Chroma or FAISS

## 2. Why LangGraph

LangGraph is a good fit because this project is not just one chatbot call. It has multiple steps that need shared state:

```text
Input text
  -> Preprocessing Agent
  -> Retrieval / Filtering Agent
  -> Keyword Extraction Agent
  -> ESG Classification Agent
  -> Insight Summary Agent
  -> Visualization Agent
  -> Final Report Agent
```

Each step can be a LangGraph node. The graph state stores intermediate results such as cleaned text, keywords, ESG scores, summaries, and chart data.

This is easier to explain in the final presentation than a black-box autonomous agent.

Official references:

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph
- LangGraph Graph API: https://docs.langchain.com/oss/python/langgraph/use-graph-api
- LangGraph durable execution: https://docs.langchain.com/oss/python/langgraph/durable-execution
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat
- LangChain OpenAI integration: https://api.python.langchain.com/en/latest/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html

## 3. Proposed Agent Roles

### 3.1 Preprocessing Agent

Purpose:

- Load raw ESG text from PDF, TXT, CSV, or existing report data.
- Normalize text.
- Remove noise such as page numbers, repeated headers, tables of contents, and broken line breaks.
- Split text into useful chunks.

Inputs:

- Raw document text.

Outputs:

- Cleaned text.
- Text chunks.
- Basic document metadata.

Possible tools:

- `pypdf`
- `pdfplumber`
- `re`
- `pandas`

### 3.2 Keyword Extraction Agent

Purpose:

- Extract high-value ESG keywords.
- Compare general word frequency with ESG-specific vocabulary.
- Produce data for word cloud and keyword tables.

Methods:

- TF-IDF
- noun phrase extraction
- keyword frequency
- optional LLM-assisted keyword grouping

Outputs:

- Top keywords.
- Keyword frequencies.
- Keyword category hints.

### 3.3 ESG Classification Agent

Purpose:

- Classify chunks into ESG dimensions:
  - Environmental
  - Social
  - Governance
  - General / Other

Optional finer labels:

- carbon emissions
- water management
- energy usage
- supply chain
- employee welfare
- diversity
- risk management
- board governance
- compliance

Approach:

- Start with rule-based keyword matching for baseline.
- Add LLM classification for better explanation.
- Store both the label and the reasoning.

Outputs:

- ESG label per chunk.
- Confidence score.
- Explanation.
- Aggregated ESG distribution.

### 3.4 Retrieval Agent

Purpose:

- Let users ask questions about the ESG report.
- Retrieve the most relevant chunks.
- Pass retrieved context to the LLM.

Example questions:

- "TSMC talks about water management in which sections?"
- "What are the main governance risks?"
- "Summarize TSMC's carbon reduction strategy."

Approach:

- MVP: simple keyword search or TF-IDF similarity.
- Upgrade: vector database with embeddings.

Possible APIs:

- OpenAI embeddings API.
- Chroma or FAISS for local vector storage.

### 3.5 Insight Summary Agent

Purpose:

- Convert analysis results into readable insights.
- Explain the strongest ESG themes.
- Compare the proportion of E, S, and G content.
- Point out interesting trends or imbalances.

Outputs:

- Executive summary.
- ESG bullet insights.
- Suggested presentation talking points.

### 3.6 Visualization Agent

Purpose:

- Prepare chart-ready data for Streamlit.
- Generate word clouds and ESG distribution charts.

Outputs:

- Word cloud image.
- ESG category counts.
- Keyword table.
- Optional timeline or section-level chart.

This agent does not need to call an LLM. It can be a deterministic Python node.

## 4. LangGraph Architecture

Recommended state shape:

```python
from typing import TypedDict, Any

class ESGState(TypedDict, total=False):
    raw_text: str
    cleaned_text: str
    chunks: list[str]
    keywords: list[dict[str, Any]]
    esg_classifications: list[dict[str, Any]]
    retrieved_chunks: list[str]
    user_question: str
    qa_answer: str
    summary: str
    chart_data: dict[str, Any]
    errors: list[str]
```

Recommended graph:

```text
START
  -> preprocess
  -> extract_keywords
  -> classify_esg
  -> build_visualizations
  -> summarize_insights
  -> END
```

For Q&A mode:

```text
START
  -> preprocess / load_cached_chunks
  -> retrieve_context
  -> answer_question
  -> END
```

## 5. API Integration Plan

### 5.1 Environment Variables

Create a `.env` file, but do not commit real keys.

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Possible future options:

```text
OLLAMA_MODEL=llama3.1
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### 5.2 OpenAI Model Calls

Recommended for this project:

- Use `langchain-openai` inside LangGraph nodes for clean integration.
- Keep model calls isolated in `agents/llm.py` so the provider can be swapped later.

Example structure:

```python
import os
from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.2,
    )
```

Then in an agent node:

```python
def summarize_insights(state: ESGState) -> ESGState:
    llm = get_llm()
    prompt = build_summary_prompt(state)
    response = llm.invoke(prompt)
    return {"summary": response.content}
```

### 5.3 OpenAI Responses API Alternative

If we want fewer LangChain dependencies, we can call the OpenAI API directly:

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Summarize the ESG themes in this text..."
)

text = response.output_text
```

Use this direct approach only if LangChain integration becomes annoying. For the main final project, LangGraph + `langchain-openai` is cleaner.

### 5.4 Embeddings API

Use embeddings only after the basic version works.

Purpose:

- Convert report chunks into vectors.
- Retrieve the most relevant chunks for user questions.

Recommended flow:

```text
chunks
  -> embeddings
  -> vector store
  -> top-k relevant chunks
  -> LLM answer
```

Possible implementation:

- `OpenAIEmbeddings`
- `Chroma`
- `FAISS`

Keep this as Phase 2 because keyword search is enough for the MVP demo.

## 6. Suggested Folder Structure

```text
tsmc-esg-wordcloud/
  app.py
  FP_PLAN.md
  requirements.txt
  .env.example
  agents/
    __init__.py
    graph.py
    state.py
    llm.py
    preprocess.py
    keywords.py
    classify.py
    retrieve.py
    summarize.py
    visualize.py
  data/
    raw/
    processed/
  outputs/
    wordclouds/
    reports/
  tests/
    test_keywords.py
    test_classify.py
```

## 7. MVP Scope

The MVP should be small but complete:

1. Load ESG text from a sample TSMC report file.
2. Clean and chunk the text.
3. Extract top keywords.
4. Classify chunks into E / S / G.
5. Generate word cloud.
6. Generate a short AI summary.
7. Show all results in Streamlit.

MVP demo screen:

- Left sidebar:
  - choose report file
  - choose model
  - run analysis button
- Main page:
  - word cloud
  - ESG distribution chart
  - top keywords table
  - AI summary
  - Q&A input box

## 8. Phase Plan

### Phase 1: Baseline Pipeline

Goal:

- Get deterministic text mining working before adding too much AI.

Tasks:

- Set up folders.
- Add `.env.example`.
- Add preprocessing functions.
- Add keyword extraction.
- Add word cloud generation.
- Update Streamlit UI.

### Phase 2: LangGraph Agents

Goal:

- Convert the pipeline into LangGraph nodes.

Tasks:

- Define `ESGState`.
- Implement graph nodes.
- Compile graph.
- Connect Streamlit button to graph invocation.
- Display intermediate outputs.

### Phase 3: LLM Summaries and ESG Classification

Goal:

- Add actual AI behavior.

Tasks:

- Add OpenAI model wrapper.
- Add LLM-based ESG classification.
- Add final insight summary.
- Add fallback behavior when no API key is present.

Fallback:

- If `OPENAI_API_KEY` is missing, use rule-based classification and show a message in the UI.

### Phase 4: Retrieval Q&A

Goal:

- Let users ask questions about the TSMC ESG report.

Tasks:

- Add simple retriever.
- Add answer generation node.
- Optional: add embeddings and vector store.

### Phase 5: Final Polish

Goal:

- Make the demo clear and presentation-ready.

Tasks:

- Improve UI layout.
- Add loading states.
- Add exported report text.
- Add sample questions.
- Add architecture diagram for presentation.
- Add tests for non-LLM functions.

## 9. Interactive Demo Layer

The final layer should not feel like a static report. It should feel like an interactive ESG intelligence dashboard.

Recommended final UI concept:

```text
Streamlit App
  -> Upload / choose report
  -> Run LangGraph analysis
  -> Explore ESG dashboard
  -> Ask questions
  -> Generate final presentation summary
```

### 9.1 Interactive Features

Core interactive features:

- Report selector: choose TSMC annual ESG report or upload a new PDF/TXT file.
- Run analysis button: triggers the LangGraph workflow.
- ESG filter tabs: view Environmental, Social, Governance, or All.
- Keyword explorer: click or select a keyword to see related chunks.
- Word cloud controls:
  - choose ESG category
  - choose top-N keywords
  - toggle Chinese / English stopwords
- Q&A box: ask questions about the report.
- Source viewer: show the report chunks used to answer the question.
- Summary mode selector:
  - executive summary
  - investor-style summary
  - classroom presentation summary
  - short demo script
- Export button: export final insights as Markdown or TXT.

Nice-to-have interactive features:

- Year comparison if multiple reports are available.
- Company comparison if we add other semiconductor ESG reports.
- Clickable ESG distribution chart.
- "Ask follow-up" suggestions generated after each answer.
- Human-in-the-loop review: user can approve or edit the AI summary before export.

### 9.2 Visual Design Direction

The app should look polished for a live demo, but still feel like an analytical tool.

Recommended style:

- Dark or semi-dark dashboard background.
- TSMC-inspired accent colors:
  - teal / cyan for Environmental
  - warm amber for Social
  - violet or blue for Governance
- Large first-screen dashboard with clear metrics.
- Plotly charts for animated hover interactions.
- Streamlit cards for:
  - total chunks analyzed
  - top ESG category
  - top keyword
  - AI confidence / coverage
- Word cloud as the main visual centerpiece.
- Use icons in section titles only if the UI remains clean.

Avoid:

- A plain vertical notebook-like page.
- Too many long paragraphs at the top.
- Showing raw LLM output without formatting.
- Making the user scroll too much before seeing results.

### 9.3 Final Screen Layout

Recommended app layout:

```text
Sidebar
  - report input
  - model selector
  - ESG category filter
  - keyword count slider
  - run analysis button

Main
  Row 1: KPI metrics
  Row 2: word cloud + ESG distribution chart
  Row 3: top keywords table + selected keyword evidence
  Row 4: AI insight summary
  Row 5: Q&A chat panel with source chunks
```

This gives the demo a strong visual opening while still showing the agent system clearly.

### 9.4 Agent-to-UI Mapping

Each agent should produce something visible in the interface:

| Agent | UI Output |
|---|---|
| Preprocessing Agent | document stats, chunks analyzed |
| Keyword Extraction Agent | word cloud, keyword table |
| ESG Classification Agent | ESG distribution chart, category filters |
| Retrieval Agent | source chunks for Q&A |
| Insight Summary Agent | final summary and talking points |
| Visualization Agent | chart-ready data and generated figures |

This is important for presentation because the audience can see each agent's contribution instead of only seeing one final answer.

## 10. Risks and Mitigations

### Risk: API key or quota issue

Mitigation:

- Keep rule-based fallback.
- Cache LLM outputs during demo.
- Use a cheaper model for summaries.

### Risk: LLM outputs inconsistent labels

Mitigation:

- Ask for structured JSON output.
- Validate labels.
- Fall back to rule-based label if parsing fails.

### Risk: PDF extraction is messy

Mitigation:

- Save extracted text to `data/processed/`.
- Manually inspect one sample.
- Use cleaned text for final demo if needed.

### Risk: Too much agent complexity

Mitigation:

- Make each agent a simple LangGraph node.
- Avoid autonomous browser/file-control agents.
- Keep the graph explainable.

## 11. Presentation Story

Recommended story:

1. Problem: ESG reports are long and hard to analyze manually.
2. Solution: Build a LangGraph multi-agent text mining system.
3. Architecture: Each agent handles one analytical task.
4. Demo: Upload TSMC ESG text, run analysis, view word cloud and ESG insights.
5. Technical value:
   - structured workflow
   - explainable intermediate steps
   - deterministic text mining + LLM reasoning
   - optional retrieval Q&A
6. Future work:
   - compare multiple years
   - compare TSMC with other semiconductor firms
   - add multilingual analysis
   - add vector search and citations

## 12. Immediate Next Steps

1. Confirm existing `app.py` structure.
2. Add `.env.example` and update `requirements.txt`.
3. Create `agents/state.py` and `agents/graph.py`.
4. Implement preprocessing and keyword extraction first.
5. Add OpenAI integration only after the baseline pipeline runs.
