# amazonlinux:2023 matches the EC2 host OS so the Duckling binary and the
# Python venv baked into the golden AMI (/opt/duckling and /opt/scheduler0-nlp)
# share the same glibc/ABI as this container.  The venv is mounted read-only
# at /opt/scheduler0-nlp at runtime; start.sh runs its uvicorn binary directly.
# We keep python3 installed so the venv's interpreter symlink and stdlib resolve.
FROM amazonlinux:2023

RUN dnf install -y python3 \
    && dnf clean all

WORKDIR /app

COPY intent.py app.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
  CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["./start.sh"]
