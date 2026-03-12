# Deploy Python backend (no Docker)

Use a **deployment service** (Render or Railway) and point it at this folder.

---

## Option 1: Render

1. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**.
2. Connect your Git repo (GitHub/GitLab).
3. Render will detect the repo’s **Blueprint** (`render.yaml` at repo root). It defines one web service with `rootDir: goalconvo-backend`.
4. After the service is created, open it → **Environment** and add your env vars (from `.env.example`), at least:
   - `MISTRAL_API_KEY` (or `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.)
   - Any other keys your pipeline uses.
5. Deploy. The backend will be at `https://<your-service>.onrender.com`. Use `/health` to check.

**Note:** Render’s free tier spins down after inactivity; first request may be slow.

---

## Option 2: Railway

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**.
2. Select this repo. When adding a **service**, set **Root Directory** to `goalconvo-backend`.
3. Railway will detect Python and use `requirements.txt` and the **Procfile** (`web: python backend_server.py`).
4. In the service → **Variables**, add the same env vars as in Render (e.g. `MISTRAL_API_KEY`, etc.).
5. Under **Settings** → **Networking**, add a **Public networking** domain so the backend is reachable.
6. Deploy. Use `/health` to verify.

---

## Production behavior

- The server reads `PORT` from the platform (Render/Railway set this automatically).
- Debug mode is off unless you set `FLASK_DEBUG=true` in the service’s environment.
- Data under `data/` and `logs/` is **ephemeral** on free tiers (lost on redeploy). For persistence, use a database or attached disk if the platform supports it.
