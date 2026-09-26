# Contributing to scheduler0-edge-classifier

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/scheduler0-edge-classifier.git
   cd scheduler0-edge-classifier
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/scheduler0/scheduler0-edge-classifier.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Docker (for running Duckling locally)
- Git
- 4GB+ RAM (for spaCy and Duckling)

### Local Environment Setup

1. **Create a Python virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Start the development server**:
   ```bash
   ./run-local.sh
   ```

   This script will:
   - Start Duckling in a Docker container
   - Launch the FastAPI app with hot-reload
   - Make the API available at http://localhost:5000

### Alternative: Manual Setup

If you prefer not to use Docker:

1. **Build Duckling** (requires Haskell Stack):
   ```bash
   ./install_scheduler0_nlp.sh
   ```

2. **Start Duckling**:
   ```bash
   cd ~/workspace/duckling
   stack exec duckling-example-exe
   ```

3. **Start the API**:
   ```bash
   source .venv/bin/activate
   uvicorn app:app --host 0.0.0.0 --port 8080 --reload
   ```

## Making Changes

### Branching Strategy

1. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits

3. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Commit Messages

Write clear commit messages following these guidelines:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests after the first line

**Examples**:
```
feat: add support for Spanish locale

- Integrate es_core_news_sm spaCy model
- Add locale detection in analyze endpoint
- Update tests for multilingual support

Closes #123
```

```
fix: correct temporal resolution for UK timezone

Duckling was using wrong reference time for BST dates.

Fixes #456
```

**Commit Prefixes**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks
- `ci:` - CI/CD changes

## Testing

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run specific test file
python -m unittest tests.test_intent -v

# Run with coverage (if configured)
coverage run -m unittest discover -s tests
coverage report
```

### Adding Tests

- Add test cases to `tests/test_intent.py` for intent classification
- Follow the existing test structure:
  - Group by expected decision (`allow`, `clarify`, `reject`)
  - Include a boolean indicating if Duckling should detect time
  - Duckling is mocked in tests for determinism

**Example**:
```python
TEST_CASES = {
    "allow": [
        ("Schedule a meeting for next Friday at 2pm", True),
        ("Remind me to call John tomorrow", True),
    ],
    # ...
}
```

### Evaluation Tools

The repository includes evaluation scripts:

```bash
# Find adversarial examples (requires Claude CLI)
./eval_adversarial.py

# Generate classifier fixes (requires Claude CLI)
./fix_classifier.py
```

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**:
   ```bash
   python -m unittest discover -s tests -v
   ```

2. **Update documentation** if needed:
   - Update README.md for new features
   - Add docstrings to new functions
   - Update API examples if endpoints changed

3. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changed and why
   - Include examples/screenshots if applicable

5. **Respond to review feedback**:
   - Make requested changes
   - Push additional commits to your branch
   - Request re-review when ready

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #(issue number)

## Changes Made
- List of changes
- Bullet points

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Added/updated tests for changes
- [ ] Updated documentation
- [ ] Code follows style guidelines
- [ ] Commits are clear and descriptive
```

## Code Style

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black default)
- Use meaningful variable names
- Add type hints to function signatures

**Example**:
```python
from typing import Dict, Any

def classify(text: str) -> Dict[str, Any]:
    """Classify user text for scheduling intent.
    
    Args:
        text: User message to classify
        
    Returns:
        Classification result with decision, reason, and features
    """
    # Implementation
    pass
```

### Code Organization

- Keep functions focused and single-purpose
- Extract complex logic into helper functions
- Use docstrings for public functions
- Group related functions together
- Avoid deep nesting (max 3-4 levels)

### Documentation Style

- Use Google-style docstrings
- Document parameters, return values, and exceptions
- Include examples for complex functions
- Keep comments up-to-date with code changes

## Documentation

### When to Update Documentation

Update documentation when you:

- Add or change API endpoints
- Modify configuration options
- Change deployment procedures
- Add new features
- Fix bugs that users might encounter

### Documentation Files

- `README.md` - Project overview and quick start
- `CONTRIBUTING.md` - This file (contribution guidelines)
- `SECURITY.md` - Security policies and reporting
- Code docstrings - Function/class documentation

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## Recognition

Contributors will be recognized in:

- Release notes for significant contributions
- GitHub contributors page
- Special acknowledgments for security reports

Thank you for contributing to scheduler0-edge-classifier! 🎉
