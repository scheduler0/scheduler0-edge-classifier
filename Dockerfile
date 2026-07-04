# amazonlinux:2023 matches the EC2 host OS so the Duckling binary (mounted from
# /opt/duckling at runtime) is guaranteed to be compatible with the container's glibc.
FROM amazonlinux:2023

RUN dnf install -y python3 \
    && dnf clean all \
    && python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip --no-cache-dir

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt \
    && python3 -m spacy download en_core_web_sm

COPY intent.py app.py start.sh ./
RUN chmod +x start.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
  CMD curl -f http://localhost:8080/healthz || exit 1

CMD ["./start.sh"]
