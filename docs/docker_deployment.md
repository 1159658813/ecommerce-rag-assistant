# P2-A Docker Desktop local demo (baseline v2.1.0 / 37bfe14)

This is a **local interview demo** deployment, not public production hosting. Do not expose
port 8000 beyond loopback: the demo API accepts user_id from the caller and does not
implement real authentication.

## Prerequisites

- Windows + Docker Desktop (Linux containers, Compose v2), sufficient disk/RAM for Torch and BGE models.
- Python 3.11 for the host-side `scripts/docker_preflight.py` (no host pip install needed).
- Root `.env` with a real `DASHSCOPE_API_KEY`. Never commit or send the key.
- `data/raw/` from Git; `data/index/knowledge.faiss` and `data/index/chunks.json` generated locally.

## 1. Prepare the branch and files

Do not overwrite the frozen `v2.1.0` tag. Apply the provided P2-A bundle on a clean
`feature/p2-docker-deployment` branch after verifying `git status`.

## 2. Build the image

```powershell
docker compose version
docker compose config --quiet
docker compose build
```

The first build may download large Python packages, including Torch. This P2 stage preserves
`requirements.txt`; avoid changing production dependency pins before a real install failure.

### Optional mirrors for restricted networks

For developers who cannot reliably access the default Debian
and PyPI repositories, the image supports configurable mirrors.

Example using USTC mirrors:

```powershell
docker compose build --build-arg DEBIAN_MIRROR=https://mirrors.ustc.edu.cn --build-arg PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
```

These build arguments do not change the Python dependency
versions in requirements.txt.

### Optional runtime proxy for Hugging Face

Some environments require a proxy to download model weights.

For Docker Desktop on Windows, configure a working proxy
in Docker Desktop settings first.

Then optionally set the following in the local .env file:

```dotenv
RAG_PROXY_URL=http://http.docker.internal:3128
```

The address and port are environment-dependent. Do not assume
they work on every computer.

With a successfully built image:

```powershell
docker compose up -d --no-build
```

The RAG_PROXY_URL setting is optional. Without it, the application
does not explicitly configure an outbound HTTP proxy.

Never commit .env, proxy credentials, or API keys.

## 3. Generate the FAISS index only if missing

For an existing, valid local `data/index/`, skip this step.

```powershell
docker compose run --rm rag-app python scripts/build_index.py
```

The command reads the bind-mounted `data/raw/` and writes `data/index/` to your **host**.
It may download an additional Qwen tokenizer for chunking. If you already have a working
host Python environment, `python scripts/build_index.py` is an alternative.

## 4. Single focused deployment validation

```powershell
python scripts/docker_preflight.py
docker compose up -d
docker compose ps
docker compose logs --tail=120 rag-app
Invoke-RestMethod http://127.0.0.1:8000/health
```

After startup, open `http://127.0.0.1:8000/` and `http://127.0.0.1:8000/docs`.
Use the UI to ask one **real** policy question and one order question, e.g.
`ORD202610002 的物流到哪了？` with demo user `U001`.
Alternatively, send a test to `POST /api/v2/chat`:

```powershell
$body = @{ question='ORD202610002 的物流到哪了？'; user_id='U001' } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v2/chat -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

The `/health` endpoint only confirms that the server reached its running state; it does not
test the DashScope API, index correctness, model output, or permissions.

## Persistence and cache

- `./data:/app/data`: existing FAISS index and SQLite database survive container replacement.
- Named Compose volume `hf-cache:/opt/hf-cache`: Hugging Face weights persist.
- Startup runs `python -m src.orders.init_db` (idempotent tables and demo seed rows).
- One Uvicorn worker avoids redundant model loading; the embedding and reranker choose CPU
  when no CUDA device is available.
- Frontend is served by FastAPI; `frontend/index.html` loads Vue/Element/Axios from `unpkg.com`
  via the user's **browser**, so a disconnected browser won't render the UI fully.

To stop without removing data/cache:

```powershell
docker compose down
```

Avoid `docker compose down -v`: it removes the persisted Hugging Face cache volume.

## Acceptance

Record: `docker compose ps`, a successful health response, screenshot of the UI,
real business query result (redact sensitive fields), first startup duration, and relevant
logs. Do not claim Docker deployment passed until these checks succeed on Docker Desktop.
Do not rerun the paid 60-case E2E suite for this infrastructure change.
