# Team Workplan: TSMC ESG LangGraph Agents

## Team Size

This project has 3 members.

Recommended strategy:

- Member A coordinates the technical workflow and app integration.
- Member B and Member C focus on project components that support the system and final presentation:
  - data preparation
  - prompt review
  - beta testing
  - presentation slides
  - demo script
  - result interpretation

This keeps the project organized while giving each member a clear contribution.

## 1. Role Split

| Role | Main Focus | Main Files |
|---|---|---|
| Member A: Technical Workflow / App Integration | LangGraph workflow, Streamlit app, API integration, text mining logic, final system integration | `agents/`, `app.py`, `requirements.txt`, `.env.example` |
| Member B: Research / Data / Testing Support | ESG background research, TSMC report preparation, prompt review, beta testing, result checking | `data/`, test notes, report sections |
| Member C: Presentation / UI Feedback / Demo Support | PPT, demo script, screenshots, UI feedback, beta testing checklist, final rehearsal | presentation materials, screenshots, test notes |

## 2. Member A: Technical Workflow / App Integration

### Responsibilities

- Define the shared `ESGState`.
- Build the LangGraph workflow.
- Connect all agent nodes together.
- Create the LLM wrapper.
- Make sure the graph can run from Streamlit.
- Handle API key loading and fallback behavior.
- Keep the technical workflow organized and easy to explain.
- Prepare small templates so teammates can help with testing and report content.

### Main Deliverables

- `agents/state.py`
- `agents/graph.py`
- `agents/llm.py`
- `agents/preprocess.py`
- `agents/keywords.py`
- `agents/classify.py`
- `agents/retrieve.py`
- `agents/visualize.py`
- Streamlit integration in `app.py`
- `.env.example`
- Updated `requirements.txt`

### Suggested Tasks

1. Create the `agents/` folder structure.
2. Define `ESGState`.
3. Create placeholder nodes for every agent.
4. Implement the baseline deterministic pipeline:
   - preprocess
   - keyword extraction
   - rule-based ESG classification
   - visualization data
5. Build a graph like this:

```text
START
  -> preprocess
  -> extract_keywords
  -> classify_esg
  -> build_visualizations
  -> summarize_insights
  -> END
```

6. Add a function like:

```python
def run_analysis(raw_text: str) -> ESGState:
    graph = build_graph()
    return graph.invoke({"raw_text": raw_text})
```

7. Make Streamlit call `run_analysis()`.
8. Add OpenAI model wrapper, but keep fallback behavior if no API key exists.
9. Add Q&A retrieval.
10. Add final demo polish after teammates finish feedback.

### Success Criteria

- The app can run the full graph without crashing.
- The dashboard is usable for final demo.
- The team can understand the high-level flow well enough to explain it.

## 3. Member B: Research / Data / Testing Support

### Responsibilities

- Prepare ESG background research.
- Collect or organize TSMC ESG report data.
- Review whether keywords and ESG classifications make sense.
- Help write non-code report sections.
- Run beta testing with a checklist.
- Optionally complete small template-based code/data tasks if comfortable.

### Main Deliverables

- ESG background notes.
- Clean sample data notes.
- Testing notes.
- Report section draft:
  - problem motivation
  - ESG background
  - dataset description
  - result interpretation
- Optional:
  - `data/sample_questions.txt`
  - `data/esg_keywords.csv`

### Suggested Tasks

1. Read the TSMC ESG / sustainability report.
2. Write a short dataset description:
   - report name
   - year
   - source
   - why it is useful for ESG text mining
3. Prepare 10-15 sample questions for the Q&A feature.
4. Review the extracted top keywords:
   - mark useful keywords
   - mark noisy keywords
   - suggest stopwords
5. Review ESG classification results:
   - check if chunks labeled E/S/G are reasonable
   - write down examples of good and bad classifications
6. Help write result interpretation for the final report.
7. Run beta testing and record issues.

### Success Criteria

- Member B can explain the ESG problem and dataset clearly.
- Member B provides useful testing feedback.
- Member B contributes visible report content.
- Optional small data files are completed if assigned.

## 4. Member C: Presentation / UI Feedback / Demo Support

### Responsibilities

- Prepare PPT slides.
- Make the project story easy to understand.
- Collect screenshots from the app.
- Give feedback on UI clarity and visual polish.
- Run beta testing before final demo.
- Prepare and rehearse the demo script.
- Optionally adjust simple Streamlit text labels if comfortable.

### Main Deliverables

- PPT slides.
- Demo script.
- App screenshots.
- Beta testing checklist.
- Final presentation speaking notes.
- Optional:
  - simple UI wording suggestions
  - color/style feedback

### Suggested Tasks

1. Create PPT structure:
   - problem
   - dataset
   - system architecture
   - agent workflow
   - demo screenshots
   - results
   - limitations and future work
2. Prepare the demo script:
   - what to click
   - what result to point out
   - what to say if API is slow
3. Give UI feedback:
   - which charts are easiest to understand
   - whether the first screen looks impressive
   - whether labels are clear
4. Test the app:
   - run with sample report
   - run without API key
   - test long text
   - test empty upload
   - test Q&A
5. Collect screenshots for slides.

### Success Criteria

- PPT is clear and visually organized.
- Demo script is ready.
- Member C can explain the dashboard and demo results.
- Testing feedback helps Member A fix the app.

## 5. Shared Interfaces

Member A coordinates the code interface. The team should understand this state format at a high level for integration and presentation.

Recommended state:

```python
class ESGState(TypedDict, total=False):
    raw_text: str
    cleaned_text: str
    chunks: list[str]
    keywords: list[dict]
    esg_classifications: list[dict]
    retrieved_chunks: list[str]
    user_question: str
    qa_answer: str
    summary: str
    chart_data: dict
    errors: list[str]
```

Recommended agent function pattern:

```python
def some_agent(state: ESGState) -> ESGState:
    # read from state
    # compute result
    return {"some_key": result}
```

For presentation, explain this as:

```text
All agents share one state object.
Each agent reads the previous results and writes its own result back.
This makes the workflow easier to debug and explain.
```

## 6. Integration Order

Recommended order:

1. Member A creates skeleton graph and baseline app.
2. Member B prepares ESG background, dataset notes, and sample questions.
3. Member C prepares PPT outline and demo story.
4. Member A implements text mining and dashboard.
5. Member B tests keyword and ESG classification quality.
6. Member C tests UI and collects screenshots.
7. Member A fixes bugs and improves demo flow.
8. All members rehearse final presentation.

## 7. Timeline

### Milestone 1: Skeleton Works

Goal:

- Streamlit can run a LangGraph workflow with placeholder outputs.

Owners:

- Member A.

### Milestone 2: Text Mining Works

Goal:

- Real report text can produce keywords, wordcloud data, and ESG labels.

Owners:

- Member A implements.
- Member B reviews quality.

### Milestone 3: Interactive Dashboard Works

Goal:

- User can select/upload report, run analysis, filter results, and ask questions.

Owners:

- Member A implements.
- Member C reviews UI and tests.

### Milestone 4: AI Layer Works

Goal:

- OpenAI summary and optional LLM ESG classification work with fallback.

Owners:

- Member A implements API wrapper.
- Member B reviews summary quality and suggests prompt improvements.

### Milestone 5: Demo Ready

Goal:

- The app is polished, tested, and ready to present.

Owners:

- Member C leads PPT and demo script.
- Member B supports result interpretation.
- Member A handles final code fixes.

## 8. Beta Testing Checklist

Test cases:

- App starts correctly.
- Sample report loads correctly.
- PDF/TXT upload works.
- Empty input shows a friendly warning.
- Long report does not crash.
- Word cloud renders.
- ESG chart renders.
- Keyword table has meaningful words.
- ESG filter changes visible outputs.
- Q&A returns an answer.
- Q&A shows related source chunks.
- App still works without `OPENAI_API_KEY`.
- API error does not crash the app.
- Demo can be completed within the time limit.

## 9. Final Presentation Contribution Map

Each member should have a clear section to explain:

| Member | Presentation Topic |
|---|---|
| Member A | LangGraph architecture, code implementation, API integration, live demo |
| Member B | ESG background, dataset, keyword/classification interpretation, testing observations |
| Member C | PPT story, dashboard walkthrough, screenshots, beta testing, limitations/future work |

## 10. Recommended Git Workflow

Simple workflow, because Member A owns most code:

```text
main
  -> feature/main-dev
  -> docs/report
  -> docs/presentation
```

Rules:

- Pull before starting work.
- Keep commits small.
- Member B and Member C should avoid editing code files unless assigned.
- Member B and Member C can focus on docs, notes, screenshots, and slides.
- Use clear commit messages:
  - `Add LangGraph skeleton`
  - `Add sample QA questions`
  - `Add beta testing notes`
  - `Update presentation slides`

## 11. Best Strategy for This 3-Person Team

The safest plan is:

- Member A coordinates the technical workflow and app integration.
- Member B helps with ESG/domain understanding, sample questions, result checking, and report writing.
- Member C helps with PPT, demo story, screenshots, UI feedback, and beta testing.

This gives everyone a clear contribution and keeps the project easy to integrate.
