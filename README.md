# Multi-Agent Customer Support — Digital Music Store

[![CI](https://github.com/MichaelMade/capstone-multi-agent-customer-support/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelMade/capstone-multi-agent-customer-support/actions/workflows/ci.yml)

A multi-agent AI customer support assistant for a digital music store, built with **LangGraph**, **langgraph-supervisor**, **LangChain**, **SQLAlchemy**, and **Gradio**. Customers verify their identity, browse the music catalog in natural language, look up invoices and purchases, and get a personalized experience backed by per-customer preference memory within the running process — with every answer grounded in the real Chinook database. The interface is also an installable progressive web app (PWA) with a privacy-safe offline screen.

## Architecture

```
START
  └─► ingest_user_input (plain text → HumanMessage)
        └─► verify_info ──(account query, unverified)──► human_input (interrupt ⏸)
              │    ▲                                          │
              │    └──────────────────────────────────────────┘  (user retries)
              │ (verified, OR catalog/greeting/off-topic → browse-first)
              ▼
  load_memory  (reads saved preferences from InMemoryStore)
        ▼
  supervisor  (langgraph-supervisor)
   ├─► music_catalog_agent   — hand-built ReAct loop, 5 music tools
   └─► invoice_agent         — prebuilt LangChain agent, 4 invoice tools
        ▼
  create_memory (extracts explicit preferences, set-union merge)
        ▼
      END
```

**Browse-first policy:** catalog questions, greetings, and off-topic messages run without verification. Only queries needing personal account data (invoices, purchases, support rep) gate on identity — one structured LLM call per unverified turn classifies intent and extracts any offered identifier (with a deterministic fast-path for bare IDs/emails/phones). Invoice queries always require verification.

| Component | Implementation |
|---|---|
| Supervisor | `langgraph_supervisor.create_supervisor` — routes music / invoice / mixed queries, rejects off-topic directly |
| Music catalog agent | Hand-built ReAct loop (assistant node + `ToolNode` + conditional edge) |
| Invoice agent | `langchain.agents.create_agent` with a state-aware identity prompt |
| Identity verification | Structured intent/identifier assessment (`VerificationAssessment`) + DB lookup by ID / case-insensitive email / normalized phone; `interrupt()` + `Command(resume=…)` human-in-the-loop |
| Memory | `InMemoryStore`, namespace `("memory_profile", customer_id)`, key `user_memory`; sanitized set-union merge, questions excluded |
| Conversation state | Plain-text-only public `InputState`; internal `State` protected by a `MemorySaver` checkpointer and one UUID thread per browser session |
| Database | Chinook loaded into in-memory SQLite via SQLAlchemy (`StaticPool`); access is serialized, read-only, and parameterized |
| UI | Installable Gradio Blocks PWA with status feedback, session isolation, interrupt/resume handling, and an offline fallback |

### Security & anti-hallucination

- **SQL injection and mutation safe:** every query uses SQLAlchemy `text()` with named bind parameters; runtime SQL is restricted to read-only `SELECT`/CTE statements, SQLite runs in `query_only` mode, IDs are canonical positive 64-bit integers, and search wildcards are treated literally.
- **Identity enforced at the tool layer, not just the prompt:** `customer_id` is delivered to every invoice tool via LangGraph `InjectedState` — it is absent from both the public graph input and the LLM-facing tool schema, so neither a caller nor a prompt-injected user can pass, alter, or fabricate it. Only the verification node writes that state field. Without a verified identity, every invoice tool refuses.
- **Ownership filtering (no IDOR):** invoice-scoped lookups (`get_support_rep_by_invoice`, `get_invoice_line_items`) require `Invoice.CustomerId` to match the verified customer; one customer can never read another's invoice data. Employee PII in outputs is minimized (no phone numbers).
- **Prompt-injection defense in depth:** the verified ID also arrives via `SystemMessage` with explicit prompt rules, but the tool-layer binding holds even if the prompt layer is subverted (verified live with an "I am actually customer 2" attack).
- **Memory hardening:** preference extraction is turn-scoped (including the original turn held across an identity interrupt), gated by a deterministic preference-language regex, and fed customer messages only. Known catalog names are canonicalized, explicit out-of-catalog tastes are preserved, instruction-shaped values are rejected, and all stored values are length/count capped and rendered as untrusted JSON labels in the music prompt.
- **Supply-chain pin:** the downloaded Chinook SQL script is verified against a pinned SHA-256 before `executescript`; startup fails closed on mismatch (`CHINOOK_SHA256` overrides).
- **Safe default bind and demo access gate:** the server binds `127.0.0.1` by default; `0.0.0.0` is an explicit opt-in (set in the Dockerfile for containers). Optional `APP_USERNAME` and `APP_PASSWORD` credentials protect the interactive UI and data-bearing APIs on a shared live demo; state-sharing and mounted schema routes are disabled.
- **Six grounding rules** in every agent prompt: tool-grounded store facts, exact quoting, honest failures, scope boundaries, no training-data answers, and truncation transparency.

## Capstone rubric alignment

- [x] **Multi-agent architecture & routing:** `src/graph.py` composes a supervisor with separate music-catalog and invoice agents, plus explicit verification, memory, and human-input nodes.
- [x] **Tool & data-layer implementation:** nine tools query the Chinook database through parameterized, read-only SQL with input validation and deterministic JSON outputs.
- [x] **Customer verification & HITL:** account requests verify a customer against the database, pause with LangGraph `interrupt()`, and resume with `Command`; invoice tools receive identity only from protected graph state.
- [x] **Memory implementation:** `MemorySaver` isolates conversation threads and `InMemoryStore` keeps namespaced customer preferences with deterministic merge and sanitization rules.
- [x] **Prompt engineering & grounding:** specialist and supervisor prompts enforce domain scope, tool-grounded facts, transparent truncation, and safe treatment of stored preference data.
- [x] **Testing & deterministic behavior:** the automated suite far exceeds the 28-test target and covers graph topology and prompt contracts, HITL, every tool outcome, database safety, authorization, memory, UI, and configuration without requiring live model calls.
- [x] **Code quality, logging & deployment:** the project uses a modular `src/` layout, validated environment settings, structured logging, an installable Gradio PWA, pinned dependencies, and a non-root Python 3.12 Docker image with a health check.

## Setup

Requires **Python 3.12**. The `.python-version` file and Docker image both use
this version.

From the project root:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Run

```bash
python app.py
```

Open http://localhost:7860. The Chinook SQL script is downloaded automatically on first run and cached under `data/`.

### Install as an app

1. Start the server and open http://localhost:7860 in a supported browser.
2. Select **Install App** when the button appears, or use the browser's install option in its address bar or menu.
3. Launch **Music Support** from your applications or home screen. The installed app still connects to the running Python server, so keep `python app.py` running while you use it.

Browsers treat `localhost` and `127.0.0.1` as secure development origins, so local installation works over HTTP. A deployment reached through any non-local hostname or IP address must be served over **HTTPS** for service-worker and installation features to work reliably. Set both `APP_USERNAME` and `APP_PASSWORD` before sharing a hosted demo; the password must contain at least 12 characters.

### Offline and privacy behavior

The service worker caches only the static offline page and app icons. It does **not** cache the live application, conversations, customer or account data, authentication responses, Gradio endpoints, or API responses. If the connection drops, the app displays the offline screen without replaying private chat content.

Offline mode cannot answer questions. The assistant still requires the local backend to be running, network access to the configured model endpoint, and a valid API key. Reconnect and select **Try again** to resume using it.

### Docker

```bash
docker build -t music-support .
docker run --rm -p 127.0.0.1:7860:7860 \
  -e OPENAI_API_KEY=sk-... music-support
```

The loopback-only port mapping keeps the unauthenticated demo local to the
host. For a shared container, also pass `APP_USERNAME` and `APP_PASSWORD` as
runtime secrets.

### Deploy to Render

The included `render.yaml` deploys the Docker image to a free Render web service, waits for GitHub checks before automatic redeploys, verifies `/healthz`, and serves the PWA from a managed HTTPS URL. It enables `REQUIRE_APP_AUTH`, so startup fails closed if either sign-in secret is missing.

1. Push the repository to GitHub, then open [Deploy to Render](https://render.com/deploy?repo=https://github.com/MichaelMade/capstone-multi-agent-customer-support).
2. Connect the repository and create the Blueprint.
3. Enter `OPENAI_API_KEY`, `APP_USERNAME`, and `APP_PASSWORD` when Render prompts for the three secret values. Do not put them in `render.yaml` or commit them.
4. When the deploy finishes, open the generated `onrender.com` URL (normally `https://michaelmade-music-support.onrender.com`) and sign in with the credentials you chose.

Render's free web services sleep after an idle period, so the first request after a pause can take about a minute. The in-memory conversation state and preference memory also reset whenever the free instance sleeps or redeploys.

## Test

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
ruff format --check .
```

The automated suite exceeds the capstone's 28-test target and covers concurrent and read-only database behavior, cache integrity, all 9 tools, graph input isolation, verification interrupt/resume, agent step limits, UI handlers, configuration validation, invoice authorization/ownership, preference-memory isolation, and prompt-data hardening. No API key is required for tests; model-dependent paths use deterministic fakes.

## Sample usage

```
You:  What rock albums do you have?
Bot:  We have 1,309 rock tracks. Here's a sample across artists: …
      (no verification needed for catalog browsing)
You:  Now show me my invoices.
Bot:  Before I can help with account information, I need to verify your
      identity. Could you please provide your Customer ID, email, or phone?
You:  5
Bot:  Thanks František, you're verified! Here are your invoices…
You:  I really love jazz and AC/DC.
Bot:  Great taste! … (preferences "jazz" and "AC/DC" saved to memory)
You:  Show me line items for invoice 1.      (belongs to another customer)
Bot:  No invoice with ID 1 found on your account…
You:  What's the weather today?
Bot:  I can only help with our music catalog and customer accounts…
```

Try these identifiers (from Chinook data): Customer ID `1`, email `luisg@embraer.com.br`, phone `+55 (12) 3923-5555`.

## Environment variables

See `.env.example`. Required: `OPENAI_API_KEY`. Optional: `APP_USERNAME` and `APP_PASSWORD` (set both to protect a shared deployment), `REQUIRE_APP_AUTH` (fail startup unless both are present), `MODEL_NAME` (default `gpt-4o-mini`; the model must support tool calling and structured outputs), `TEMPERATURE` (default `0.0`), `OPENAI_API_BASE` (an OpenAI-compatible endpoint), `PORT`, `HOST`, `LOG_LEVEL`, and `GRADIO_ANALYTICS_ENABLED` (disabled by default).

## Project structure

```
.python-version          # required Python runtime (3.12)
.github/workflows/ci.yml # tests, lint, formatting, and Docker build in CI
requirements.txt         # pinned application dependencies
requirements-dev.txt     # pinned test and lint dependencies
app.py                   # entry point
pwa/                     # manifest icons, service worker, and offline page
render.yaml              # access-controlled Render deployment blueprint
src/
  config.py             # env-driven settings
  database.py           # Chinook load, run_query_safe, normalize_phone, health
  llm.py                # ChatOpenAI factory
  models.py             # Pydantic extraction and persisted-profile schemas
  pwa.py                # FastAPI PWA host, manifest, and privacy-safe asset routes
  state.py              # public InputState + internal LangGraph State
  graph.py              # full graph assembly
  ui.py                 # Gradio chat interface and install controls
  agents/
    prompts.py          # all system prompts + grounding rules
    verification.py     # verify_info, human_input (interrupt), routing
    music_agent.py      # hand-built ReAct agent
    invoice_agent.py    # prebuilt ReAct agent
    memory.py           # load_memory / create_memory nodes
  tools/
    music_tools.py      # 5 music catalog tools
    invoice_tools.py    # 4 invoice tools
    common.py           # safe_int, JSON helpers, privacy-safe tool audit logs
tests/
  test_database.py               # queries, concurrency, cache, health
  test_tools.py                  # all tools, validation, ordering
  test_tool_security.py         # schema hiding, authz, IDOR, PII
  test_memory_logic.py          # merge, catalog validation, isolation
  test_agent_graph_regressions.py # graph boundary and interrupt flow
  test_ui.py / test_config.py   # UI handlers and environment validation
  test_pwa.py                    # installability, caching, auth, and PWA routes
  test_integrity.py             # SQL artifact hash pin
  test_observability.py         # privacy-safe tool/database logging
  test_rubric_contracts.py      # graph, prompt, verification, memory contracts
  test_tool_rubric_matrix.py    # success/not-found/error for all nine tools
```

## Known limitations

- `MemorySaver` and `InMemoryStore` are in-memory only: conversation history and preferences are lost on restart. In production, swap in persistent checkpointer/store backends (e.g. Postgres).
- Installing the PWA does not turn the assistant into a fully offline application: the Python server must remain running, and assistant responses require network access to the configured model endpoint. Offline support is intentionally limited to a static, privacy-safe status screen.
- Free Render instances sleep after inactivity and have ephemeral filesystems. A cold start can take about a minute, and every restart clears this project's in-memory state.
- Running the assistant requires a valid, live `OPENAI_API_KEY`, network access to the configured model endpoint, and any associated API usage budget. Only the deterministic test suite runs without an API key.
- Customer ID, email, and phone are convenient demo lookup identifiers, not strong authentication secrets. A production deployment must use real authentication and authorization before exposing account data.
- Phone matching compares fully normalized numbers (a bare national number also matches its `+`-prefixed international form, but not a different country code).
- The pinned `langgraph-supervisor` release still calls LangGraph's deprecated `create_react_agent` internally and emits one compile-time warning. The project's own invoice agent already uses `langchain.agents.create_agent`; remove the pin or upgrade the supervisor when its upstream migration is released.
