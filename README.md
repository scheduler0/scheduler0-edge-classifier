# scheduler0 Edge Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7-blue.svg)](https://spacy.io/)

A lightweight, deterministic NLP-based intent classification service for [Scheduler0](https://github.com/scheduler0). It analyzes user messages to determine whether they represent scheduling requests, require clarification, or should be rejected before entering the main scheduler workflow.

The service exposes a FastAPI REST API and includes a conversation analysis engine that extracts commitments, requests, deadlines, and follow-ups from multi-message threads.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Local Development](#local-development)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **🎯 Intent Classification**: Determines if text is a scheduling request (`allow`), ambiguous (`clarify`), or not schedule-related (`reject`)
- **💬 Conversation Analysis**: Extracts structured suggestions (commitments, requests, deadlines) from multi-message threads
- **🌐 Temporal Understanding**: Uses Facebook's Duckling for robust time/date parsing
- **🧠 NLP-Powered**: Leverages spaCy for linguistic feature extraction
- **⚡ Lightweight & Fast**: Optimized for low-latency edge classification
- **🔍 Explainable**: Returns detailed reasoning and linguistic features with each decision
- **🐳 Production-Ready**: Includes Docker support, health checks, and ECS deployment workflows

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Docker (for Duckling time parser)
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/scheduler0/scheduler0-edge-classifier.git
cd scheduler0-edge-classifier

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start the service (includes Duckling via Docker)
./run-local.sh
```

The API will be available at `http://localhost:5000`

### Test It Out

```bash
# Health check
curl http://localhost:5000/healthz

# Classify a scheduling request
curl -X POST http://localhost:5000/v1/intents/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Remind me every Monday at 9am"}'
```

## API Reference

### Intent Classification

**Endpoint**: `POST /v1/intents/classify`

**Request**:
```json
{
  "text": "Remind me every Monday at 9am"
}
```

**Response**:
```json
{
  "text": "Remind me every Monday at 9am",
  "decision": "allow",
  "reason": "request_with_temporal_signal",
  "features": {
    "has_temporal_signal": true,
    "looks_like_request": true,
    "has_negation": false
  },
  "duckling_entities": [...],
  "tokens": [...]
}
```

**Decisions**:
- `allow` - Clear scheduling request with temporal signal
- `clarify` - Schedule-like but ambiguous (missing time or unclear intent)
- `reject` - Not a scheduling request (questions, statements, etc.)

### Conversation Analysis

**Endpoint**: `POST /v1/suggestions/analyze`

Analyzes multi-message conversations to extract structured obligations and suggestions.

**Request**:
```json
{
  "conversation_id": "conv_123",
  "messages": [
    {
      "id": "msg_1",
      "speaker": "Alice",
      "timestamp": "2026-09-26T10:00:00-04:00",
      "message": "I'll send you the proposal tomorrow"
    }
  ],
  "participants": [
    {"id": "user_1", "display_name": "Alice", "timezone": "America/Toronto"}
  ],
  "options": {
    "locale": "en_US",
    "default_timezone": "America/Toronto"
  }
}
```

**Response**: Returns structured suggestions with commitments, requests, deadlines, and follow-ups.

See the full API documentation in the source code docstrings.

## How It Works

The classifier combines multiple NLP techniques for accurate intent detection:

1. **spaCy Parsing** (`en_core_web_sm`): Extracts grammatical structure, subjects, verbs, dependencies
2. **Duckling Time Parsing**: Identifies temporal expressions (dates, times, durations, recurrences)
3. **Regex Patterns**: Detects request markers ("can you", "please"), recurrence ("every Monday"), negation
4. **Rule Engine**: Combines features using deterministic decision logic

### Classification Logic

```
Text: "Can you send me a digest every Friday?"

spaCy detects:
  - Interrogative structure
  - Second-person subject ("you")
  - Action verb ("send")

Duckling detects:
  - Recurrence pattern ("every Friday")

Regex detects:
  - Request marker ("Can you")

→ Decision: ALLOW (request + temporal signal)
```

### Suggestions Engine

The conversation analysis engine:
- Tracks obligations across multiple messages
- Detects commitments ("I'll do X"), requests ("Can you do Y"), and deadlines
- Handles state changes (completion, cancellation, rescheduling)
- Resolves ambiguous references (pronouns, temporal expressions)
- Provides confidence scoring and explainability

## Local Development

### Manual Setup (Without Docker)

If you prefer to run Duckling outside Docker:

```bash
# Install Haskell Stack
curl -sSL https://get.haskellstack.org/ | sh

# Run the full bootstrap (includes Duckling build)
./install_scheduler0_nlp.sh

# Start Duckling
cd ~/workspace/duckling
stack exec duckling-example-exe &

# Start the API
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Development Tools

```bash
# Run tests
python -m unittest discover -s tests -v

# Find adversarial examples (requires Claude CLI)
./eval_adversarial.py

# Generate fixes for failing tests (requires Claude CLI)
./fix_classifier.py
```

## Deployment

### AWS ECS Deployment

This service is designed to run on AWS ECS with dedicated EC2 capacity. See the comprehensive [DEPLOYMENT.md](DEPLOYMENT.md) guide for:

- Infrastructure setup (ECS, ECR, IAM, VPC)
- EC2 bootstrap process
- Task definition configuration
- GitHub Actions CI/CD setup
- Production considerations

### GitHub Actions

Automated deployments are configured for:
- **Staging**: Triggered on push to `staging` branch
- **Production**: Triggered on push to `main` branch

Required GitHub secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (variable)

See [.github/workflows/](.github/workflows/) for workflow details.

## Testing

### Running Tests

The test suite includes table-driven test cases grouped by expected decision (`allow`, `clarify`, `reject`):

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_intent -v
```

**Note**: Duckling is mocked in tests for deterministic results. Real spaCy parsing and decision logic are exercised.

### Evaluation Tools

The repository includes tools for finding edge cases and automatically generating fixes:

```bash
# Find adversarial examples (requires Claude CLI)
./eval_adversarial.py --target allow --max-rounds 10

# Generate and apply fixes for failing tests (requires Claude CLI)
./fix_classifier.py
```

When `eval_adversarial.py` finds a misclassification, it offers to add it as a regression test. The `fix_classifier.py` script then generates a minimal patch to fix the issue.

## Architecture

### Runtime Model

The production deployment uses a hybrid architecture:

```
EC2 Host                          ECS Container
┌─────────────────┐              ┌────────────────────┐
│ Duckling        │◄─────────────┤ FastAPI App        │
│ (Port 8000)     │   HTTP       │ - Intent Classifier│
│                 │              │ - Suggestions API  │
│ /opt/duckling/  │              │                    │
│ /opt/scheduler0-│──mounted─────►│ /duckling          │
│ nlp/            │   read-only  │ /opt/scheduler0-nlp│
└─────────────────┘              └────────────────────┘
```

**Key Design Decisions**:
- Duckling runs on the EC2 host (not in containers) due to Haskell GHC RTS compatibility
- Python virtualenv and NLP models are pre-installed on the host and mounted read-only
- This keeps container images small (~50MB) and enables fast cold starts

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete architecture details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Testing requirements
- Pull request process

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure tests pass (`python -m unittest discover -s tests -v`)
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Security

For security concerns, please see [SECURITY.md](SECURITY.md) for our vulnerability disclosure policy.

**Do not** report security vulnerabilities through public GitHub issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **spaCy**: Fast and production-ready NLP library
- **Duckling**: Robust time/date parsing from Facebook
- **FastAPI**: Modern, fast web framework for Python
- **The Scheduler0 Team**: For building intelligent scheduling automation

## Links

- [Deployment Guide](DEPLOYMENT.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [License](LICENSE)
- [Issue Tracker](https://github.com/scheduler0/scheduler0-edge-classifier/issues)

---

Made with ❤️ by the Scheduler0 team
