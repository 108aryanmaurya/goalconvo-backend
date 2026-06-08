# GoalConvo

GoalConvo is a framework for generating large-scale **goal-oriented dialogue data** using **multi-agent LLM simulation**. A User agent and a Support agent take turns in realistic task-focused conversations (e.g. booking a hotel, customer support, education). Generated dialogues are quality-filtered, stored as structured datasets, and can be evaluated with comprehensive metrics.

This repository is a **monorepo** with two main parts:

| Component | Technology | Default URL |
|-----------|------------|-------------|
| **Backend** | Python, Flask, Flask-SocketIO | http://localhost:5000 |
| **Frontend** | Next.js 15, React 19, TypeScript | http://localhost:3000 |

---

## What the System Does

1. **Experience generation** — Builds a scenario from a seed goal: user persona, context, and opening message (with few-shot examples from a curated hub).
2. **Multi-agent simulation** — User and Support agents alternate turns with optional planning, memory, and reflection.
3. **Quality filtering** — Rule-based checks plus LLM scoring (coherence, goal relevance, overall quality).
4. **Dataset storage** — Accepted dialogues saved under domain folders; versions and few-shot hub updated over time.
5. **Evaluation** — Separate pass for metrics such as goal completion, task success, diversity, BLEU/BERTScore (with reference data), and optional LLM-as-judge.

---

## Prerequisites

### For local development

| Requirement | Version |
|-------------|---------|
| **Python** | 3.8+ (3.11 recommended) |
| **Node.js** | 18+ (20 recommended) |
| **npm** | Comes with Node.js |
| **Git** | Any recent version |

### For Docker

| Requirement | Version |
|-------------|---------|
| **Docker** | 20.10+ |
| **Docker Compose** | v2+ |

### API access (required for generation)

At least **one** LLM provider API key must be configured. Supported providers (priority order for legacy single-client mode):

OpenRouter → Groq → DeepSeek → Ollama (local) → Gemini → OpenAI → Mistral/Together

You can also route **generation** and **evaluation** to different providers via environment variables (see [Environment configuration](#environment-configuration)).

### Optional

- **MultiWOZ reference data** — Needed only for BLEU/BERTScore comparison against real dialogues. Download via the backend script after setup.
- **GPU** — Not required; LLM calls go to external APIs (or local Ollama).

---

## Project Structure

```
monorepo-goalconvo/
├── goalconvo-backend/          # Python framework + REST/WebSocket server
│   ├── backend_server.py       # Main API server (port 5000)
│   ├── scripts/                # CLI: generate, evaluate, download data
│   ├── src/goalconvo/          # Core library (agents, pipeline, evaluation)
│   ├── data/                   # Synthetic dialogues, few-shot hub, versions
│   ├── requirements.txt
│   ├── setup.sh
│   └── start_backend.sh
├── goalconvo-frontend/         # Next.js dashboard
│   ├── app/                    # UI pages and components
│   ├── lib/                    # API client config
│   └── package.json
├── docker-compose.yml          # Run backend + frontend in containers
└── README.md                   # This file
```

---

## Installation

Choose **one** of the following methods.

---

### Method 1: Local development (recommended)

#### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd monorepo-goalconvo
```

#### Step 2 — Backend setup

```bash
cd goalconvo-backend
```

**Option A — Automated setup script**

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Check Python version (3.8+)
- Install the package in editable mode (`pip install -e .`)
- Install dependencies from `requirements.txt`
- Create `data/` and `logs/` directories
- Copy `.env.example` → `.env` if `.env` does not exist

**Option B — Manual setup**

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

mkdir -p data/{multiwoz,synthetic,few_shot_hub,results,versions} logs
cp .env.example .env
```

#### Step 3 — Configure API keys

Edit `goalconvo-backend/.env`:

```bash
nano .env   # or use your preferred editor
```

**Minimum:** set at least one provider key. Examples:

```env
# Example: Google Gemini for generation
GEMINI_API_KEY=your_gemini_api_key_here
GENERATION_PROVIDER=gemini

# Example: Claude via OpenRouter for evaluation
OPENROUTER_API_KEY=your_openrouter_api_key_here
EVALUATION_PROVIDER=claude
```

Other supported variables (uncomment as needed):

```env
GROQ_API_KEY=
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
MISTRAL_API_KEY=
OLLAMA_ENABLED=false
OLLAMA_API_BASE=http://localhost:11434
```

#### Step 4 — Verify backend installation

```bash
source venv/bin/activate
python -c "from goalconvo import Config; print('OK')"
python scripts/generate_dialogues.py --test-connection
```

#### Step 5 — Frontend setup

Open a **new terminal**:

```bash
cd goalconvo-frontend
npm install
```

Create environment file for the frontend:

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:5000" > .env.local
```

#### Step 6 — (Optional) Download MultiWOZ reference data

From `goalconvo-backend` with venv activated:

```bash
python scripts/download_multiwoz.py
```

Used for BLEU/BERTScore during comprehensive evaluation. Other metrics still run without it.

---

### Method 2: Docker

#### Step 1 — Clone and configure

```bash
git clone <repository-url>
cd monorepo-goalconvo
```

Create a `.env` file in the **repository root** (or ensure API keys are passed to Docker). At minimum, set keys used by the backend, for example:

```env
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
OLLAMA_ENABLED=false
```

The backend container also reads keys from `goalconvo-backend/.env` when mounted; for Docker Compose, root `.env` is interpolated into `docker-compose.yml`.

#### Step 2 — Build and start

```bash
docker compose up --build
```

Services:

| Service | Port | Description |
|---------|------|-------------|
| `goalconvo-backend` | 5000 | Flask API + WebSocket |
| `goalconvo-frontend` | 3000 | Next.js production build |

Data and logs are persisted via volumes:

- `./goalconvo-backend/data` → `/app/data`
- `./goalconvo-backend/logs` → `/app/logs`

#### Step 3 — Health check

```bash
curl http://localhost:5000/health
```

Expected response includes `"status": "healthy"`.

---

## Running the Application

### Local (two terminals)

**Terminal 1 — Backend**

```bash
cd goalconvo-backend
source venv/bin/activate
./start_backend.sh
# Or: python backend_server.py
```

Server listens on **http://localhost:5000**. First startup may take 30–60 seconds while dependencies load.

**Terminal 2 — Frontend**

```bash
cd goalconvo-frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

### Docker

```bash
docker compose up
```

Then open **http://localhost:3000**.

---

## Using the Dashboard

1. Open the frontend at http://localhost:3000.
2. **Run Pipeline** — Starts dialogue generation; live turns stream over WebSocket.
3. **Run Evaluation** — Runs comprehensive metrics on generated (or uploaded) dialogues.
4. Explore **versions**, **human evaluation**, and per-stage views in the UI.

The frontend connects to the backend using `NEXT_PUBLIC_API_URL` (default `http://localhost:5000`).

---

## Command-Line Usage (Backend)

Activate the virtual environment first:

```bash
cd goalconvo-backend
source venv/bin/activate
```

### Test LLM connection

```bash
python scripts/generate_dialogues.py --test-connection
```

### Generate dialogues

```bash
python scripts/generate_dialogues.py --num-dialogues 100 --domains hotel
```

Common flags:

| Flag | Description |
|------|-------------|
| `--num-dialogues N` | Number of dialogues to generate (default: 1000) |
| `--domains hotel restaurant` | Domains to use (default from config) |
| `--resume` | Resume from saved progress |
| `--test-connection` | Test API only, no generation |
| `--run-evaluation` | Run comprehensive evaluation after generation |

### Comprehensive evaluation

```bash
python scripts/comprehensive_dialogue_evaluation.py
```

Set `EVAL_SKIP_LLM_JUDGE=1` in `.env` to skip extra LLM judge calls and reduce API usage.

---

## Environment Configuration

Key variables in `goalconvo-backend/.env`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `GENERATION_PROVIDER` | LLM provider for dialogue generation | `gemini` |
| `EVALUATION_PROVIDER` | LLM provider for quality judge / evaluation | `claude` |
| `TEMPERATURE` | Sampling temperature | `0.7` |
| `TOP_P` | Nucleus sampling | `0.9` |
| `MIN_TURNS` / `MAX_TURNS` | Dialogue length bounds | `6` / `15` |
| `QUALITY_THRESHOLD` | Minimum score to accept a dialogue | `0.7` |
| `FEW_SHOT_EXAMPLES` | Examples shown during experience generation | `4` |
| `DATA_DIR` | Root data directory | `./data` |
| `EVAL_SKIP_LLM_JUDGE` | Set to `1` to disable LLM judge in evaluation | `0` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

Frontend (`goalconvo-frontend/.env.local`):

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | Backend base URL | `http://localhost:5000` |

---

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/run-pipeline` | Start dialogue generation (use WebSocket for live updates) |
| POST | `/api/run-evaluation` | Run comprehensive evaluation |
| GET | `/api/versions` | List dataset versions |
| GET | `/api/docs` | API documentation UI |

**WebSocket events** (join room with `session_id`):

- Pipeline: `pipeline_start`, `step_start`, `step_data`, `live_dialogue`, `log`, `pipeline_complete`, `pipeline_error`
- Evaluation: `evaluation_complete`, `evaluation_error`

---

## Supported Domains

Default domains follow task-oriented dialogue settings (e.g. **hotel**; additional domains such as restaurant, taxi, train, attraction can be enabled in configuration). The dashboard and CLI accept a domain list when starting a run.

---

## Troubleshooting

### Backend does not start

- Ensure venv is activated and dependencies installed: `pip install -r requirements.txt && pip install -e .`
- Check Python version: `python3 --version` (3.8+)
- Verify port 5000 is free: `lsof -i :5000`

### Frontend cannot reach backend

- Confirm backend is running: `curl http://localhost:5000/health`
- Check `NEXT_PUBLIC_API_URL` in `goalconvo-frontend/.env.local`
- Restart the frontend after changing `.env.local`

### Pipeline fails / no dialogues generated

- Run `python scripts/generate_dialogues.py --test-connection`
- Ensure at least one API key is set in `goalconvo-backend/.env`
- Check `goalconvo-backend/logs/` for errors

### Docker backend unhealthy

- Wait for `start_period` (40s) on first boot
- Confirm API keys are set in environment or `.env`
- Inspect logs: `docker compose logs backend`

### Slow first backend startup

- Loading ML-related dependencies (e.g. transformers, torch) can take up to a minute on first run; this is normal.

---

## Development

### Run backend tests

```bash
cd goalconvo-backend
source venv/bin/activate
pytest
```

### Frontend lint

```bash
cd goalconvo-frontend
npm run lint
```

### Production frontend build

```bash
cd goalconvo-frontend
npm run build
npm start
```

---

## License

MIT License — see LICENSE file if present in the repository.

## Citation

If you use GoalConvo in research, please cite the associated GoalConvo paper when available.
