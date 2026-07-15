# scheduler0 Edge Classifier

Intent classification service for Scheduler0. It decides whether a user message is a scheduling request, needs clarification, or should be rejected before the main scheduler workflow continues.

The service is exposed as a small FastAPI app and is deployed to ECS on dedicated EC2 capacity.

## What It Does

`POST /v1/intents/classify` returns:

- `allow` when the text looks like a scheduling request with a temporal signal.
- `clarify` when the text is schedule-like but ambiguous.
- `reject` when the text does not look like a scheduling request.

The classifier combines:

- spaCy dependency parsing with `en_core_web_sm`.
- Duckling time and duration parsing.
- Lightweight request, recurrence, question, and negation regexes.

## API

Health check:

```bash
curl http://localhost:8080/healthz
```

Classify text:

```bash
curl -X POST http://localhost:8080/v1/intents/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Remind me every Monday at 9am"}'
```

Example response:

```json
{
  "text": "Remind me every Monday at 9am",
  "decision": "allow",
  "reason": "request_with_temporal_signal",
  "features": {
    "has_temporal_signal": true,
    "looks_like_request": true
  }
}
```

## Runtime Model

The ECS task runs an Amazon Linux 2023 container, but the heavier NLP dependencies are installed on the EC2 host:

- Duckling is built and exported under `/opt/duckling`.
- The Python virtualenv is created under `/opt/scheduler0-nlp`.
- ECS mounts both directories read-only into the container.
- `start.sh` discovers the EC2 host private IP, waits for Duckling, then starts Uvicorn from the mounted virtualenv.

This keeps the container small and avoids rebuilding Duckling inside the image.

## Local Development

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Duckling must be reachable at `http://127.0.0.1:8000/parse` for classification to work locally.

Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Run the classifier smoke examples:

```bash
python intent.py
```

## Docker

Build the container:

```bash
docker build -t scheduler0-intent-classifier .
```

The production container expects the host-mounted paths from the ECS task definition:

- `/duckling`
- `/opt/scheduler0-nlp`

For plain local Docker runs, prefer running Uvicorn directly from a local Python environment unless you also provide compatible mounts.

## EC2 Bootstrap

`install_scheduler0_nlp.sh` prepares the dedicated classifier EC2 instance:

```bash
./install_scheduler0_nlp.sh --ci
```

It installs system packages, builds Duckling, creates the NLP virtualenv, downloads the spaCy model, writes a bootstrap marker, and exports runtime files for ECS mounting. Cold bootstrap can take 30 to 90 minutes.

## ECS Deployment

The staging deployment workflow is triggered by pushes to the `staging` branch:

```text
.github/workflows/deploy-staging.yml
```

The reusable workflow builds the image, pushes it to ECR, renders `staging-scheduler0-intent-classifier-ecs-task-definition.json`, registers a new task definition, and updates the ECS service.

The task definition pins the service to dedicated classifier capacity:

```json
{
  "type": "memberOf",
  "expression": "attribute:scheduler0.intent-classifier == true"
}
```

The matching Terraform user data configures the EC2 instance with:

```text
ECS_INSTANCE_ATTRIBUTES={"scheduler0.intent-classifier":"true"}
```

## Useful Troubleshooting

Check ECS service status:

```bash
aws ecs describe-services \
  --cluster staging_scheduler0_ecs_cluster \
  --services staging_scheduler0-intent-classifier-service
```

Check the EC2 bootstrap log:

```bash
tail -f /var/log/scheduler0-intent-classifier-bootstrap.log
```

Verify bootstrap marker:

```bash
cat /var/lib/scheduler0-intent-classifier/bootstrap-complete
```

Verify Duckling:

```bash
/opt/duckling/duckling-example-exe --help
```
