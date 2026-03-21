# IAAS

**Intelligent Academic Advising System** is a Penn State-focused advising platform that combines a Custom multi-agent orchestration built on top of LangChain model and retrieval primitives, React frontend, a lightweight Node authentication service, and a Flask-based RAG AI orchestration engine for transcript-aware academic guidance.

The project is built around a practical AI engineering pattern: use LLMs for reasoning, planning, and synthesis, while keeping critical academic logic deterministic, observable, and tool-driven.

Hosted on AWS : https://3.145.213.72/ 

## Overview

IAAS supports:

- transcript upload and session-scoped retrieval
- semester-by-semester graduation planning
- next-semester course recommendations
- Penn State policy and deadline lookup
- course, major, and prerequisite question answering
- source-backed web research for current or unexpected factual questions
- faculty-facing major roadmap graph generation

## Why This Project Is Interesting

This codebase is not a basic chatbot wrapper. It demonstrates:

- multi-agent orchestration with explicit planning, execution, and validation stages
- hybrid AI systems design combining LLM reasoning with deterministic academic logic
- retrieval-augmented generation over transcripts, courses, majors, and policy data
- tool-aware execution with controlled web research and source validation
- operational observability through per-request debug traces and regression tests

## System Architecture

```text
Browser
  -> React Client
      -> Flask AI API
          -> Orchestrator Agent
          -> Executor Agent
          -> Finalizer Agent
          -> Shared Research Pipeline
          -> Local RAG Stores
          -> Deterministic Graduation Planner
      -> Node / Express Auth API
          -> MySQL
```

### Main Services

#### 1. React frontend

Location: `IAAS/client`

Responsibilities:

- renders the student advising chat experience
- manages per-tab session state
- uploads transcript PDFs
- sends conversation history with each request
- surfaces faculty roadmap generation views

#### 2. Node authentication service

Location: `IAAS/client/server`

Responsibilities:

- register and login endpoints
- MySQL user storage
- lightweight auth support for the UI

#### 3. Flask AI backend

Location: `IAAS/LLMAcadamic-Advisor`

Responsibilities:

- transcript ingestion and retrieval setup
- multi-agent reasoning pipeline
- deterministic graduation planning
- policy and deadline lookup
- web-backed research for fresh facts
- debug trace generation

## AI Multi-Agentic Workflow

The active AI workflow is a three-stage agent pipeline with a shared research system behind it.

### Agent 1: Orchestrator

Primary responsibility:

- understand the user message in context
- correct spelling and grammar when useful
- resolve follow-up questions against prior conversation
- decide what type of answer is needed
- choose whether tools are required
- produce a structured execution plan

Inputs:

- latest user message
- conversation history
- current server-local date/time
- active task context from prior turns

Output:

- `OrchestrationPlan`

What it decides:

- domain: transcript, course, major, policy, or general academic chat
- deliverable: direct answer, audit, single-semester plan, multi-semester plan, full graduation plan, or lookup result
- plan scope and semester horizon
- allowed tools
- executor path
- success criteria and failure constraints

### Agent 2: Executor

Primary responsibility:

- execute the orchestrator plan exactly
- call tools only when needed
- combine deterministic outputs and LLM reasoning into a grounded draft

Core executor paths:

- `conversation_profile_recall`
- `conversation_plan_slice`
- `policy_lookup`
- `research_lookup`
- transcript-, course-, major-, and planning-related handlers

Tools available to the executor:

- conversation history and active task context
- transcript retriever
- course retriever
- majors retriever
- policy retriever
- deterministic graduation planner
- prerequisite graph utilities
- shared web research pipeline

### Agent 3: Finalizer

Primary responsibility:

- rewrite the executor draft into a polished user-facing response
- keep the answer friendly, professional, and appropriately detailed
- validate that the draft matches the original plan
- check citations, freshness, answer scope, and date requirements
- trigger one controlled retry if the draft fails validation

Validation examples:

- a single-semester request must not return a full graduation plan
- a current deadline question should not return without a concrete or clearly exhausted source-backed answer
- a no-tools conversation-only request must not silently call tools

### Shared Research Pipeline

The web research layer is shared infrastructure rather than a top-level agent, but it is a critical part of the system.

Responsibilities:

- resolve research context from the question and conversation
- infer campus, term, year, freshness needs, and exact-date requirements
- choose a search strategy
- retrieve evidence from official sources first
- rank, fetch, and extract candidate facts
- return a source-backed research result to the executor

Provider ladder:

1. official Penn State seed URLs
2. official domain crawl
3. DuckDuckGo discovery
4. Tavily fallback when configured
5. browser fallback for difficult pages
6. deterministic claim extraction
7. LLM synthesis over verified evidence

### Inner Agent Communication

The internal handoff between stages is explicit.

1. `Orchestrator -> Executor`
   Sends a structured `OrchestrationPlan` with the normalized question, requested scope, allowed tools, executor path, and success criteria.

2. `Executor -> Finalizer`
   Sends an `ExecutorResult` containing the draft response, tools used, execution metadata, and any search resolution artifacts.

3. `Finalizer -> Executor` (optional)
   Sends a single retry reason if validation fails, for example scope mismatch, missing concrete date, or missing source.

4. `System -> Debug Trace`
   Records orchestrator, executor, finalizer, tool usage, research rounds, quality checks, and repair attempts.

## Retrieval-Augmented Generation (RAG)

IAAS uses RAG to ground responses in academic data instead of relying on free-form model memory alone.

### What is retrieved

The system retrieves from four main knowledge sources:

- uploaded transcript PDFs for transcript-aware advising
- `processed_psu_courses.csv` for course information and prerequisites
- `all_abington_majors_combined(5).csv` for major requirements
- `PennState_Combined_Advising_and_Registrar_VERBATIM.pdf` for policy and registrar guidance

### How the RAG pipeline works

1. A transcript is uploaded through `/llm/setup-environment`.
2. The backend hashes the file and creates a session-scoped transcript collection.
3. The transcript is loaded, chunked, lightly redacted, embedded, and stored in Chroma.
4. Static academic sources such as courses, majors, and policy documents are also indexed in local vector stores.
5. At response time, the executor retrieves the most relevant chunks for the current question.
6. The retrieved context is passed into the response-building stage so the answer is grounded in local evidence.

### Why RAG matters here

RAG improves:

- accuracy for transcript-specific advising
- consistency for course and major questions
- grounded policy responses from local Penn State material
- explainability, because answers are tied to retrievable academic context

### RAG and the rest of the system

RAG is one layer of the architecture, not the whole system.

- for static academic knowledge, the system prefers local retrieval first
- for planning, retrieval is combined with deterministic graduation logic
- for current or time-sensitive facts, the system escalates from local RAG to the shared web research pipeline

This design keeps the system both grounded and adaptable: local retrieval handles stable university knowledge, while web research handles freshness-sensitive questions that embeddings alone cannot answer reliably.

## Engineering Highlights

- **Hybrid reasoning model**: LLMs handle planning and synthesis; deterministic code handles transcript parsing, prerequisite ordering, and schedule generation.
- **Tool-aware orchestration**: tool usage is explicit, constrained, and validated after execution.
- **Search for both expected and unexpected questions**: the system can route known policy questions and open-ended factual questions through the same research pipeline.
- **Session continuity**: follow-up questions like `current semester` or `what about Abington` can continue the active reasoning thread.
- **Performance-conscious web research**: the search pipeline uses staged query expansion, bounded concurrency, short-lived caches, and early exits when an authoritative answer is found.
- **Observability**: each response can return a detailed internal trace for debugging and evaluation.

## Key Technical Components

### Frontend

- React 18
- React Router
- MUI
- Axios

### Backend

- Flask
- Flask-CORS
- Express
- MySQL

### AI / Retrieval

- OpenAI `gpt-5.4-mini`
- LangChain
- Chroma
- PDF ingestion with `pypdf`
- local CSV- and PDF-backed retrieval corpora
- RAG

### Research / Verification

- `duckduckgo-search`
- `tavily-python` when configured
- `requests`
- Playwright fallback for difficult pages

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── IAAS/
│   ├── client/
│   │   ├── src/
│   │   ├── server/
│   │   └── package.json
│   ├── LLMAcadamic-Advisor/
│   │   ├── api/
│   │   │   └── api.py
│   │   ├── LLM/
│   │   │   ├── main_copy.py
│   │   │   ├── orchestration_pipeline.py
│   │   │   ├── web_search_pipeline.py
│   │   │   ├── graduationFlowchart.py
│   │   │   ├── graphExperiment.py
│   │   │   ├── majorsGraph.py
│   │   │   ├── processed_psu_courses.csv
│   │   │   ├── all_abington_majors_combined(5).csv
│   │   │   └── PennState_Combined_Advising_and_Registrar_VERBATIM.pdf
│   │   └── requirements.txt
│   ├── test_orchestration_pipeline.py
│   └── test_web_search_pipeline.py
└── QUICK_START.md
```



## API Endpoints

### Flask AI API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/llm/setup-environment` | upload a transcript and initialize retrieval |
| `POST` | `/llm/generate-response` | generate the main advising response |
| `POST` | `/llm/clear-conversation` | clear in-memory conversation and active research context |
| `POST` | `/llm/generate-major-graph` | generate a major roadmap graph |

### Node API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/register` | register a user |
| `POST` | `/api/login` | authenticate a user |

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- MySQL if you want auth endpoints enabled
- OpenAI API access

### 1. Install Python dependencies

The Flask advising service has its own dependency file and should be treated as the authoritative environment for the AI backend.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r IAAS/LLMAcadamic-Advisor/requirements.txt
```

Optional browser fallback support:

```bash
playwright install chromium
```

### 2. Install frontend and auth dependencies

```bash
cd IAAS/client
npm install
cd server
npm install
cd ..
```

### 3. Configure environment

Recommended variables:

```bash
export OPENAI_API_KEY=...
export AGENT_DEBUG_TRACE=true
export REACT_APP_LLM_URL=http://127.0.0.1:5002/llm
export MYSQL_DISABLE=true
export TAVILY_API_KEY=...
```

`MYSQL_DISABLE=true` is useful if you want to run the advising system without the auth database.

### 4. Start the application

```bash
cd IAAS/client
npm start
```

This starts:

- frontend on `http://localhost:3000`
- Node auth service on `http://localhost:5001`
- Flask AI service on `http://127.0.0.1:5002`

## Testing

### Targeted AI pipeline tests

```bash
python3 -m unittest IAAS/test_orchestration_pipeline.py IAAS/test_web_search_pipeline.py
```

### Manual conversation smoke test

```bash
python IAAS/test_ai_conversation.py
```

The Flask service must already be running for the manual conversation test.

## Current Status

This repository is a strong engineering prototype and an excellent AI systems portfolio project. It is not fully production-hardened yet.

Known gaps:

- some configuration and secret handling still need cleanup
- some state is still process-local rather than fully externalized
- the auth layer is lightweight and needs stronger production hardening
- there are multiple dependency manifests in the repo and they should be consolidated over time


## Summary

IAAS is a practical example of modern AI application engineering:

- a real frontend and backend stack
- explicit multi-agent orchestration
- deterministic planning for high-stakes academic logic
- source-aware research for fresh information
- traceable and testable behavior instead of opaque prompt-only answers

It is the kind of project that shows both software engineering discipline and applied AI systems thinking.

## Core Files To Read First

If you want to understand the system quickly, start here:

1. `IAAS/client/src/components/UnifiedChat.jsx`
2. `IAAS/client/src/services/apiService.js`
3. `IAAS/LLMAcadamic-Advisor/api/api.py`
4. `IAAS/LLMAcadamic-Advisor/LLM/orchestration_pipeline.py`
5. `IAAS/LLMAcadamic-Advisor/LLM/main_copy.py`
6. `IAAS/LLMAcadamic-Advisor/LLM/web_search_pipeline.py`
7. `IAAS/LLMAcadamic-Advisor/LLM/graduationFlowchart.py`
