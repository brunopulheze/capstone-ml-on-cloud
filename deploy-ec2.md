EC2 deployment (minimal, low-cost)

1) Build the Docker image locally (or on the EC2 instance)

```bash
# from project root
docker build -t btc-predictor:latest .
```

2) Run locally for smoke test

```bash
docker run --rm -p 8080:8080 \
  -e MODEL_DIR=/app/models \
  -e MODEL_TYPE=auto \
  -v "$(pwd)/models:/app/models" \
  btc-predictor:latest
```

3) EC2 (recommend t3.small or t3.micro for demo)
- Create an EC2 instance (Amazon Linux 2 or Ubuntu) and open port 8080 in the security group.
- Install Docker on the instance.
- Copy the image (via docker save -> scp -> docker load) or push to Docker Hub/ECR and pull on the instance.
- Ensure `models/` is copied to `/home/ec2-user/app/models` (or mount an EBS volume).
- Run the same `docker run` command on EC2.

4) Startup script (systemd) example

Create `/etc/systemd/system/btc-predictor.service`:

```
[Unit]
Description=BTC Predictor Container

[Service]
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/bin/docker run --rm -p 8080:8080 \
  -e MODEL_DIR=/app/models \
  -e MODEL_TYPE=auto \
  -v /home/ec2-user/app/models:/app/models \
  --name btc-predictor btc-predictor:latest
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable btc-predictor
sudo systemctl start btc-predictor
```
