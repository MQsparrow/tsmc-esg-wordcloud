# Contribution Guide

This guide explains the concrete tasks for each team member and how the work connects to the final project.

The goal is to make sure the system is not only working in code, but also clear, credible, well-tested, and easy to present.

## Member A: Technical Workflow and App Integration

Member A focuses on the technical workflow and app integration.

### Main Goals

- Build the LangGraph agent workflow.
- Connect the workflow to the Streamlit dashboard.
- Coordinate text mining, ESG classification, visualization, and Q&A modules.
- Add OpenAI API integration with fallback behavior.
- Prepare a stable demo version with the team.

### Checklist

- Create the `agents/` folder structure.
- Define the shared state in `agents/state.py`.
- Build the LangGraph workflow in `agents/graph.py`.
- Add model/API handling in `agents/llm.py`.
- Implement preprocessing, keyword extraction, ESG classification, retrieval, and visualization.
- Update `app.py` so users can run analysis from the Streamlit interface.
- Make sure the app works even if `OPENAI_API_KEY` is missing.
- Fix issues found during beta testing.
- Prepare the final deployed version.

## Member B: ESG Research and Result Review

Member B focuses on the ESG content, dataset quality, and result interpretation.

### Main Goals

- Make sure the project has a clear ESG motivation.
- Prepare useful reference information about the TSMC ESG report.
- Help evaluate whether the system output makes sense.
- Provide sample questions and result interpretation for the report.

### Checklist

- Find the exact TSMC ESG / sustainability report source.
- Record the report title, year, and source link.
- Write a short explanation of why ESG reports are useful for text mining.
- Prepare 10-15 sample questions for the Q&A feature.
- Review the generated top keywords.
- Mark useful keywords.
- Mark noisy or meaningless keywords that should be removed.
- Review ESG classification examples.
- Pick 3 examples where the classification works well.
- Pick 2 examples where the classification is unclear or incorrect.
- Write a short interpretation of the final results.

### Suggested Sample Questions

Examples:

- What are the main environmental topics in the report?
- How does TSMC discuss water management?
- What does the report say about carbon reduction?
- What social responsibility topics appear most often?
- What governance risks are mentioned?
- Which ESG category appears most frequently?
- What are the most important keywords in the report?
- How does TSMC describe supply chain responsibility?
- What are the major sustainability goals?
- What sections are related to employee welfare?

## Member C: Presentation, Demo, and Testing

Member C focuses on presentation quality, demo flow, screenshots, and usability testing.

### Main Goals

- Make the final story clear and easy to follow.
- Prepare a polished PPT.
- Make sure the demo flow is reliable.
- Test the app from a user's point of view.

### Checklist

- Create the PPT outline.
- Prepare slides for:
  - problem motivation
  - dataset
  - system architecture
  - LangGraph agent workflow
  - dashboard demo
  - results
  - limitations
  - future work
- Make an architecture slide using this flow:

```text
Input ESG report
  -> Preprocessing Agent
  -> Keyword Extraction Agent
  -> ESG Classification Agent
  -> Visualization Agent
  -> Insight Summary Agent
  -> Streamlit Dashboard
```

- Take screenshots of the final Streamlit dashboard.
- Write the demo script.
- Decide the order of clicks during the demo.
- Prepare backup explanation if the API is slow.
- Run beta testing and record issues.
- Help rehearse the final presentation.

## Beta Testing Checklist

Use this checklist before the final demo.

- App starts correctly.
- Sample report loads correctly.
- PDF or text upload works.
- Empty input shows a clear warning.
- Long report does not crash the app.
- Word cloud renders correctly.
- ESG chart renders correctly.
- Top keyword table appears.
- ESG filter changes the displayed result.
- Q&A returns a useful answer.
- Q&A shows related source chunks.
- App works without `OPENAI_API_KEY`.
- API errors do not crash the app.
- Dashboard looks clear on the presentation screen.
- Demo can be completed within the time limit.

## Demo Script Template

Use this structure for the live demo.

```text
1. Open the Streamlit app.
2. Select or upload the TSMC ESG report.
3. Click Run Analysis.
4. Show the KPI metrics.
5. Explain the word cloud.
6. Show the ESG distribution chart.
7. Select one ESG category filter.
8. Show the keyword table.
9. Ask one sample question in the Q&A box.
10. Show the answer and source chunks.
11. End with the AI-generated summary.
```

## Report Writing Suggestions

Useful sections for the final report:

- Project motivation.
- Dataset description.
- System architecture.
- Agent roles.
- Text mining methods.
- ESG classification method.
- Dashboard design.
- Results and observations.
- Limitations.
- Future work.

## Final Presentation Contribution Map

| Member | Main Speaking Part |
|---|---|
| Member A | LangGraph workflow, app integration, API integration |
| Member B | ESG background, dataset, keyword and classification interpretation |
| Member C | Dashboard walkthrough, demo script, testing, presentation results |
