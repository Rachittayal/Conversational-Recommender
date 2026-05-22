# SHL Assessment Recommender

A conversational AI agent that helps hiring managers discover and select the right SHL assessments for their roles. Built with FastAPI, Groq LLM, and semantic search capabilities.

## Overview

This system acts as an intelligent assistant that understands hiring needs through natural conversation and recommends appropriate assessments from SHL's product catalog. Instead of browsing through hundreds of assessments manually, users can describe their role requirements and get personalized recommendations in seconds.

### What It Does

- **Conversational Interface**: Engages in natural dialogue to understand hiring requirements
- **Context Extraction**: Automatically identifies role, seniority, domain, and assessment needs from conversation
- **Semantic Search**: Uses FAISS and sentence transformers to find relevant assessments beyond keyword matching
- **Smart Recommendations**: Proactively suggests complementary assessments (e.g., adding personality tests for leadership roles)
- **Comparison Support**: Explains differences between specific assessments when asked
- **Refinement**: Allows users to add or remove items from recommendations iteratively

## Architecture

The system is built around a multi-stage pipeline that minimizes LLM calls while maintaining conversational quality:

### Core Components

#### 1. **Agent (`agent.py`)**
The orchestration layer that manages conversation flow and decision routing.

**Key Design Decisions:**
- **Zero-cost early exits**: Deterministic checks for off-topic queries and finality signals before calling the LLM
- **Templated clarification**: Uses rule-based questions for common scenarios (missing role/seniority) to avoid unnecessary LLM calls
- **Single analysis call**: One LLM invocation extracts all context, decides action, and generates clarifying questions
- **Proactive enrichment**: Automatically adds OPQ32r for leadership roles and Verify G+ for cognitive-heavy positions

**Conversation Actions:**
- `RECOMMEND`: Build and return assessment shortlist
- `CLARIFY`: Ask targeted questions to gather missing context
- `COMPARE`: Explain differences between specific assessments
- `REFINE`: Modify existing recommendations (add/remove items)
- `CONFIRM`: Acknowledge user satisfaction and end conversation
- `REFUSE`: Politely decline off-topic requests

#### 2. **Retriever (`retriever.py`)**
Handles semantic search and filtering using FAISS vector similarity.

**Search Pipeline:**
1. Embed query using `sentence-transformers/all-MiniLM-L6-v2`
2. Find top-K semantically similar items via FAISS inner product search
3. Boost items with exact name matches in query
4. Apply hard filters (test type, job level, language, remote capability)
5. Progressive relaxation if no results (drop language → job level → remote filters)

**Why This Approach:**
- Semantic search catches conceptual matches ("leadership assessment" → OPQ32r)
- Name boosting ensures exact requests surface first
- Filter relaxation prevents empty results while respecting user intent

#### 3. **Prompts (`prompts.py`)**
Contains all LLM prompt templates with detailed instructions.

**Critical Prompt: `build_analysis_prompt`**
This single prompt does the heavy lifting:
- Decides which action to take based on conversation state
- Extracts all context slots from full conversation history
- Generates grounded clarifying questions (not generic slot-fillers)
- Detects refinement requests and comparison intents
- Identifies finality signals

**Why One Prompt:**
Multiple sequential LLM calls are slow and expensive. This prompt reasons holistically about all decisions at once.

#### 4. **Catalog (`catalog.py`)**
Loads and caches the SHL product catalog from remote JSON.

**Features:**
- Network fetch with local caching to `data/catalog.json`
- Maps human-readable test types to internal codes (A, B, C, D, E, K, P, S)
- Validates required fields and handles missing data gracefully

#### 5. **Schemas (`schemas.py`)**
Pydantic models for type safety and validation.

**Key Models:**
- `ChatRequest/ChatResponse`: API contract
- `CatalogItem`: Immutable product representation with embedding text generation
- `ConversationAnalysis`: Structured output from analysis prompt
- `ContextSlots`: Extracted user requirements (role, seniority, test types, etc.)

#### 6. **Config (`config.py`)**
Centralized configuration with environment variable support.

**Configurable:**
- Groq API key and model selection
- Catalog URL and cache path
- Conversation limits (max turns, max clarifying questions)
- Search parameters (top-K, max recommendations)
- Test type mappings

## Technical Highlights

### 1. Efficiency Optimizations

**Deterministic Fast Paths:**
- Off-topic detection uses keyword matching before LLM (zero cost)
- Finality detection checks for satisfaction signals in short responses (zero cost)
- Templated clarification for common missing fields (zero cost)
- Minimum context check prevents premature LLM analysis

**Result:** Most conversations use 2-3 LLM calls instead of 5-7.

### 2. Context Extraction

The analysis prompt extracts context from the **entire conversation history**, not just the last message. This handles:
- Users correcting themselves ("Actually, make it senior level")
- Information spread across multiple turns
- Implicit signals (e.g., "15+ years experience" → senior)

### 3. Grounded Clarification

Instead of generic questions like "What seniority level?", the system asks contextual questions:
- "Is this backend-leaning (Java/Spring heavy) or a true full-stack role with significant Angular work?"
- "Is this for selecting new hires or developing existing leaders?"

This is achieved by instructing the LLM to reference what it already knows.

### 4. Progressive Filter Relaxation

When hard filters produce no results, the retriever relaxes constraints in priority order:
1. Drop language requirements
2. Drop job level requirements  
3. Drop remote requirements
4. Return pure semantic matches

This prevents "no results" responses while respecting user intent as much as possible.

### 5. Proactive Enrichment

For leadership roles, the system automatically adds OPQ32r (personality) if not present. For cognitive-heavy roles (graduate, technical), it adds Verify G+ (cognitive ability). This mimics expert consultant behavior.

## API Endpoints

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a senior Java developer"},
    {"role": "assistant", "content": "Is this for backend or full-stack?"},
    {"role": "user", "content": "Backend, Spring Boot primarily"}
  ]
}
```

**Response:**
```json
{
  "reply": "For a senior backend Java developer, I'm including Java 8 for technical skills and OPQ32r for personality assessment.",
  "recommendations": [
    {
      "name": "Java 8",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/...",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

### `GET /health`

Returns service status. Returns 503 during startup or if initialization failed.

## Setup

### Prerequisites

- Python 3.11+
- Groq API key ([get one here](https://console.groq.com/))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Assessment_SHL
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# or if using uv:
uv sync
```

4. Create `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Running the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive API docs: `http://localhost:8000/docs`

## Configuration

Edit `config.py` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Required: Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model to use |
| `MAX_TURNS` | 8 | Maximum conversation length |
| `MAX_CLARIFY_QUESTIONS` | 2 | Max questions before forcing recommendation |
| `TOP_K_RESULTS` | 20 | Semantic search candidates |
| `MAX_RECOMMENDATIONS` | 10 | Maximum items in shortlist |

## Project Structure

```
Assessment_SHL/
├── agent.py              # Conversation orchestration and routing
├── retriever.py          # Semantic search with FAISS
├── prompts.py            # LLM prompt templates
├── catalog.py            # Product catalog loading
├── schemas.py            # Pydantic models
├── config.py             # Configuration management
├── main.py               # FastAPI application
├── data/
│   └── catalog.json      # Cached product catalog
├── .env                  # Environment variables (not in git)
├── pyproject.toml        # Project dependencies
└── README.md
```

## Design Philosophy

### 1. Minimize LLM Calls
Every LLM call adds latency and cost. We use deterministic logic wherever possible and consolidate multiple decisions into single prompts.

### 2. Fail Gracefully
- Progressive filter relaxation prevents empty results
- Fallback to templated questions if LLM analysis fails
- Generic error responses instead of exposing internal errors

### 3. Ground in Data
- Never invent assessment details
- All recommendations must exist in catalog
- Comparisons only use provided catalog data

### 4. Conversational Quality
- Ask targeted questions, not generic forms
- Reference previous context naturally
- Detect when user is satisfied and end gracefully

## Limitations and Future Work

**Current Limitations:**
- No multi-language support (English only)
- No user authentication or session persistence
- Limited to SHL assessment domain
- No feedback loop for recommendation quality

**Potential Improvements:**
- Add conversation history persistence (database)
- Implement user feedback collection
- Support multi-turn refinement with undo/redo
- Add assessment comparison matrix visualization
- Integrate with SHL's actual product APIs for real-time data
- Support batch recommendations for multiple roles

## Testing

Run the test suite:
```bash
pytest
```

For manual testing, use the interactive docs at `/docs` or tools like curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I need to hire a senior data scientist"}
    ]
  }'
```

