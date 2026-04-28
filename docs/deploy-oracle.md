# Oracle Cloud Deployment — BTC Predictor API

**Stack**: Docker · FastAPI · scikit-learn Random Forest  
**Platform**: Oracle Cloud Infrastructure (OCI) — Always Free tier  
**Instance**: VM.Standard.A1.Flex (ARM Ampere) — up to 4 OCPUs + 24 GB RAM, permanently free  
**Image registry**: Docker Hub (`brunopulheze/btc-predictor`)

---

## Prerequisites

- [Oracle Cloud account](https://cloud.oracle.com) — sign up with a credit card (won't be charged; Always Free never expires)
- Docker Hub image already pushed: `brunopulheze/btc-predictor:latest` ✅
- SSH key pair generated locally (see Step 2)

---

## Step 1 — Generate SSH key pair (local)

```powershell
ssh-keygen -t rsa -b 4096 -f "$HOME\.ssh\oci-btc-key" -N '""'
```

This creates:
- `C:\Users\bruno\.ssh\oci-btc-key`      ← private key (keep secret)
- `C:\Users\bruno\.ssh\oci-btc-key.pub`  ← public key (paste into OCI console)

---

## Step 2 — Launch an Always Free VM

1. Log in to **[cloud.oracle.com](https://cloud.oracle.com)** → **Compute → Instances → Create Instance**

2. **Name**: `btc-predictor`

3. **Image**: Click **Change image** → select **Canonical Ubuntu 22.04** (or 24.04)

4. **Shape**: Click **Change shape**:
   - Shape series: **Ampere** (ARM)
   - Shape: `VM.Standard.A1.Flex`
   - OCPUs: `1` | Memory: `6 GB` *(well within the 4 OCPU / 24 GB always-free quota)*

5. **SSH keys**: select **Paste public key** → paste the contents of `oci-btc-key.pub`:
   ```powershell
   Get-Content "$HOME\.ssh\oci-btc-key.pub"
   ```

6. Leave everything else as default → click **Create**

7. Wait ~1–2 min until **State = Running**, then note the **Public IP address**

---

## Step 3 — Open port 8080 in OCI firewall

OCI has two layers of firewall — both must allow port 8080.

### 3a — Security List (OCI network layer)

1. Click the instance → **Subnet** link → **Security Lists** → **Default Security List**
2. **Add Ingress Rules** → fill in:
   | Field | Value |
   |-------|-------|
   | Source CIDR | `0.0.0.0/0` |
   | IP Protocol | `TCP` |
   | Destination Port | `8080` |
3. Click **Add Ingress Rules**

### 3b — OS firewall (inside the VM)

After SSHing in (Step 4), run:

```bash
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
sudo netfilter-persistent save
```

---

## Step 4 — SSH into the instance

```powershell
$OCI_IP = "<YOUR-PUBLIC-IP>"   # replace with actual IP
ssh -i "$HOME\.ssh\oci-btc-key" ubuntu@$OCI_IP
```

Fix key permissions if SSH refuses the key:

```powershell
icacls "$HOME\.ssh\oci-btc-key" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

---

## Step 5 — Install Docker on the VM

Once SSH'd in, run:

```bash
sudo apt-get update -y
sudo apt-get install -y docker.io netfilter-persistent iptables-persistent
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
exit   # re-login so group change takes effect
```

SSH back in:

```bash
ssh -i "$HOME\.ssh\oci-btc-key" ubuntu@$OCI_IP
```

---

## Step 6 — Pull and run the container

```bash
# Open port 8080 in the OS firewall
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
sudo netfilter-persistent save

# Pull and start the container
docker pull brunopulheze/btc-predictor:latest

docker run -d \
  --name btc-predictor \
  --restart unless-stopped \
  -p 8080:8080 \
  -e MODEL_DIR=/app/models \
  brunopulheze/btc-predictor:latest
```

Check logs to confirm startup:

```bash
docker logs -f btc-predictor
# Should see: INFO: Application startup complete.
```

---

## Step 7 — Verify from your local machine

```powershell
$OCI_IP = "<YOUR-PUBLIC-IP>"
Invoke-RestMethod http://${OCI_IP}:8080/health
Invoke-RestMethod http://${OCI_IP}:8080/
Invoke-RestMethod http://${OCI_IP}:8080/predict/latest | ConvertTo-Json
```

Expected:

```json
{ "status": "healthy" }
{ "status": "ok", "model": "RF", "n_features": 25 }
{ "predicted_price": 75212.93, "previous_close": 75872.52, "last_data_date": "2026-04-22", "data_points": 4236 }
```

Full smoke test:

```powershell
python tests/smoke_test.py --url http://${OCI_IP}:8080
```

---

## Step 8 — (Optional) systemd auto-restart on reboot

```bash
sudo tee /etc/systemd/system/btc-predictor.service > /dev/null <<EOF
[Unit]
Description=BTC Predictor Docker Container
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a btc-predictor
ExecStop=/usr/bin/docker stop btc-predictor

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable btc-predictor
```

---

## Cost

| Resource | Cost |
|----------|------|
| VM.Standard.A1.Flex (1 OCPU, 6 GB RAM) | **$0 — Always Free, never expires** |
| 50 GB boot volume | $0 (Always Free includes 200 GB total) |
| Outbound data (first 10 GB/month) | $0 |

> OCI Always Free resources **do not require stopping** to avoid charges — they are genuinely free indefinitely.
