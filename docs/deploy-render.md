# Render Deployment — BTC Predictor API

**Stack**: Docker · FastAPI · TensorFlow/Keras GRU  
**Platform**: [Render](https://render.com) (free tier)  
**Live URL**: https://btc-predictor-ngio.onrender.com  
**Image registry**: Docker Hub (`brunopulheze/btc-predictor`)

---

## Prerequisites

- Docker Desktop running locally (image `btc-predictor:latest` already built)
- Docker Hub account (`brunopulheze`) — image already pushed
- [Render](https://render.com) account (free, GitHub login works)

---

## Step 1 — Push image to Docker Hub

```powershell
# Log in (prompts for password)
docker login

# Tag the local image
docker tag btc-predictor:latest brunopulheze/btc-predictor:latest

# Push
docker push brunopulheze/btc-predictor:latest
```

> Already done — `brunopulheze/btc-predictor:latest` is live on Docker Hub.

---

## Step 2 — Create a Web Service on Render

1. Go to **[render.com](https://render.com)** → **New → Web Service**
2. Select **"Deploy an existing image from a registry"**
3. **Image URL**: `docker.io/brunopulheze/btc-predictor:latest`
4. Fill in:

   | Field | Value |
   |-------|-------|
   | Name | `btc-predictor` |
   | Region | Frankfurt (EU Central) |
   | Instance type | **Free** |

5. **Environment Variables** → Add:
   - `PORT` = `8080`

6. **Health Check Path** → `/health`

7. Click **Deploy Web Service**

Render pulls the image and deploys it (~2–3 min). The public URL is assigned automatically.

> **Free tier caveat**: the service spins down after 15 min of inactivity. The first request after a cold start takes ~30 s.

---

## Step 3 — Verify

```powershell
$URL = "https://btc-predictor-ngio.onrender.com"
Invoke-RestMethod "$URL/health"
Invoke-RestMethod "$URL/"
```

Expected responses:

```json
{ "status": "healthy" }
{ "status": "ok", "model": "GRU", "lookback": 20 }
```

Full predict smoke test:

```powershell
python tests/smoke_test.py --url https://btc-predictor-ngio.onrender.com
```

---

## Redeploying after model changes

1. Retrain the GRU by running `notebooks/bitcoin-price-prediction.ipynb`
2. Rebuild and push the Docker image:
   ```powershell
   docker build -t btc-predictor:latest .
   docker tag btc-predictor:latest brunopulheze/btc-predictor:latest
   docker push brunopulheze/btc-predictor:latest
   ```
3. In the Render dashboard → **Manual Deploy → Deploy latest image**

---

## Cost estimate

| Tier | RAM | Cost |
|------|-----|------|
| Free | 512 MB | $0/month (spins down after inactivity) |
| Starter | 512 MB | $7/month (always on) |
| Standard | 2 GB | $25/month (recommended for TensorFlow) |

> The free tier has 512 MB RAM. TensorFlow startup uses ~400–500 MB — it works but is tight. Upgrade to Starter ($7/month) for always-on reliability.
