# amazonlinux:2023 matches the EC2 host OS so the Python venv baked into the
# golden AMI (/opt/scheduler0-nlp) shares the same glibc/ABI as this container.
# The venv is mounted read-only at /opt/scheduler0-nlp at runtime; start.sh
# runs its uvicorn binary directly.  We keep python3 so the venv's interpreter
# symlink and stdlib resolve.
#
# Duckling runs on the EC2 HOST (not inside this container); the container
# reads the host IP from /duckling/host_ip (mounted from /opt/duckling) and
# waits for Duckling to be available before starting Uvicorn.
FROM amazonlinux:2023

RUN dnf install -y python3 curl-minimal \
    && dnf clean all

WORKDIR /app

COPY intent.py app.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
  CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["./start.sh"]
