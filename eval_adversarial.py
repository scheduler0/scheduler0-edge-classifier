#!/usr/bin/env python3
"""Find an adversarial example for the local intent classifier using Claude."""

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DECISIONS = ("allow", "clarify", "reject")
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_CLAUDE_PATH = Path.home() / ".local/bin/claude"
DEFAULT_TEST_PATH = Path(__file__).parent / "tests/test_intent.py"
TEST_INSERTION_MARKER = (
    "    # eval_adversarial.py inserts confirmed cases above this marker.\n"
)

CLASSIFIER_DESCRIPTION = """
The classifier assigns exactly one decision:
- allow: a clear scheduling request containing a time or recurrence.
- clarify: schedule-like text whose request or time is ambiguous, including
  negated schedule-like requests.
- reject: text that is not a scheduling request, including informational
  questions and declarative statements about schedules.

Its implementation uses these imperfect heuristics:
- A temporal signal is a Duckling time/duration entity or the regex
  "every|each" followed by a weekday/day/week/month/year/time-of-day.
- A request is detected from polite phrases anywhere in the text ("can you",
  "please", "let's", "could we", "I'd like to schedule", "I want a reminder",
  "how about" plus a scheduling verb), from a sentence-initial command, or
  when the root verb has no subject. Only the root's subject counts.
- Text starting with what/why/how/when/where/who/which is rejected as an
  informational question, except scheduling proposals such as "how about we
  meet" and "when should we meet".
- Asks whose root verb is tell/explain/describe are rejected even if they
  mention a time, unless they complement a scheduling verb ("tell me to book").
- don't/do not/dont/never/stop/cancel/remove plus a temporal signal clarifies.
  "stop by", "don't forget", "don't let me forget", and discourse "never mind"
  are not treated as canceling the request.
- Decision order: informational question or informational request -> reject;
  negation plus temporal -> clarify; temporal plus declarative statement ->
  reject; temporal plus request -> allow; temporal only -> clarify; request
  only -> clarify; otherwise -> reject.
- French text is detected from diacritics, clock times like "9h", or a French
  lexicon, then classified with the same order using French request, question,
  negation, recurrence, and statement patterns. English spaCy tags are not
  trusted for French. Duckling is called with locale fr_FR for that text.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Claude to find a local intent-classifier misclassification."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"classifier base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--claude-path",
        type=Path,
        default=DEFAULT_CLAUDE_PATH,
        help=f"Claude Code executable (default: {DEFAULT_CLAUDE_PATH})",
    )
    parser.add_argument(
        "--target",
        choices=DECISIONS,
        help="only generate examples whose correct decision is this category",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="maximum Claude generation rounds (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="candidates requested per Claude round (default: 5)",
    )
    parser.add_argument(
        "--claude-timeout",
        type=int,
        default=180,
        help="seconds allowed for each Claude invocation (default: 180)",
    )
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error("--max-rounds must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def check_prerequisites(
    base_url: str, claude_path: Path, session: requests.Session
) -> None:
    if not claude_path.is_file():
        raise RuntimeError(f"Claude Code CLI not found at {claude_path}")
    try:
        response = session.get(f"{base_url}/healthz", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"classifier is not healthy at {base_url}; run ./run-local.sh first"
        ) from exc


def build_generation_prompt(
    target: Optional[str],
    batch_size: int,
    attempts: List[Dict[str, str]],
) -> str:
    target_instruction = (
        f'Every candidate must have the human-correct label "{target}".'
        if target
        else "Use any of allow, clarify, and reject as the human-correct label."
    )
    if attempts:
        attempt_json = json.dumps(attempts[-50:], ensure_ascii=False, indent=2)
        feedback = f"""
These previous candidates were classified correctly and therefore failed to
break the classifier. Do not repeat them or make merely cosmetic variants:
{attempt_json}
"""
    else:
        feedback = "There are no previous attempts."

    return f"""
You are adversarially evaluating a scheduling intent classifier. Generate
exactly {batch_size} realistic user messages that exploit likely gaps between
human intent and the heuristic implementation. The goal is for the classifier
to return a decision different from the objectively correct decision.

{CLASSIFIER_DESCRIPTION}

{target_instruction}
{feedback}

Return ONLY a JSON array. Each element must have exactly these string fields:
{{"text": "the user message", "expected": "allow|clarify|reject"}}
The expected field is the human-correct decision, not the decision you predict
the implementation will produce. Do not include Markdown or commentary.
""".strip()


def extract_json_array(output: str) -> List[Any]:
    text = output.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end <= start:
            raise ValueError("Claude response did not contain a JSON array")
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Claude returned invalid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ValueError("Claude response must be a JSON array")
    return value


def generate_candidates(
    claude_path: Path,
    prompt: str,
    target: Optional[str],
    timeout: int,
) -> List[Dict[str, str]]:
    try:
        result = subprocess.run(
            [
                str(claude_path),
                "-p",
                prompt,
                "--output-format",
                "text",
                "--tools",
                "",
                "--no-session-persistence",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Claude timed out after {timeout}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Claude CLI failed: {detail or exc}") from exc

    candidates: List[Dict[str, str]] = []
    for index, item in enumerate(extract_json_array(result.stdout), start=1):
        if not isinstance(item, dict):
            print(f"  Skipping candidate {index}: not an object", file=sys.stderr)
            continue
        text = item.get("text")
        expected = item.get("expected")
        if not isinstance(text, str) or not text.strip():
            print(f"  Skipping candidate {index}: invalid text", file=sys.stderr)
            continue
        if expected not in DECISIONS:
            print(
                f"  Skipping candidate {index}: invalid expected decision",
                file=sys.stderr,
            )
            continue
        if target and expected != target:
            print(
                f"  Skipping candidate {index}: expected is not target {target}",
                file=sys.stderr,
            )
            continue
        candidates.append({"text": text.strip(), "expected": expected})
    if not candidates:
        raise ValueError("Claude returned no valid candidates")
    return candidates


def classify(
    base_url: str, text: str, session: requests.Session
) -> Dict[str, Any]:
    response = session.post(
        f"{base_url}/v1/intents/classify",
        json={"text": text},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("decision") not in DECISIONS:
        raise ValueError("classifier returned an invalid decision")
    return payload


def print_success(
    round_number: int, candidate: Dict[str, str], result: Dict[str, Any]
) -> None:
    print("\nSUCCESS: found a classifier failure")
    print(f"Round:    {round_number}")
    print(f"Text:     {candidate['text']}")
    print(f"Expected: {candidate['expected']}")
    print(f"Actual:   {result['decision']}")
    print(f"Reason:   {result.get('reason', '(none)')}")


def confirm(message: str) -> bool:
    try:
        answer = input(f"{message} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def add_regression_test(
    test_path: Path,
    candidate: Dict[str, str],
    result: Dict[str, Any],
) -> bool:
    source = test_path.read_text(encoding="utf-8")
    if TEST_INSERTION_MARKER not in source:
        raise RuntimeError(f"test insertion marker not found in {test_path}")

    tree = ast.parse(source, filename=str(test_path))
    existing_prompts = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    text = candidate["text"]
    if text in existing_prompts:
        print(f"\nRegression prompt is already present in {test_path}.")
        return False

    has_time = bool(result.get("features", {}).get("duckling_has_time"))
    entry = f"    ({text!r}, {candidate['expected']!r}, {has_time!r}),\n"

    print("\nProposed regression test:")
    print(f"  Text:              {text}")
    print(f"  Expected decision: {candidate['expected']}")
    print(f"  Duckling has time: {has_time}")
    if not confirm(f"Add this failing case to {test_path}?"):
        print("Regression test was not added.")
        return False

    updated = source.replace(TEST_INSERTION_MARKER, entry + TEST_INSERTION_MARKER, 1)
    temporary_path = test_path.with_suffix(".py.tmp")
    temporary_path.write_text(updated, encoding="utf-8")
    temporary_path.replace(test_path)
    print(f"Added regression test to {test_path}.")
    print("Run ./fix_classifier.py to ask Claude for a classifier patch.")
    return True


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    session = requests.Session()

    try:
        check_prerequisites(base_url, args.claude_path.expanduser(), session)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    attempts: List[Dict[str, str]] = []
    seen = set()

    for round_number in range(1, args.max_rounds + 1):
        print(f"\nRound {round_number}/{args.max_rounds}: asking Claude for candidates...")
        prompt = build_generation_prompt(args.target, args.batch_size, attempts)
        try:
            candidates = generate_candidates(
                args.claude_path.expanduser(),
                prompt,
                args.target,
                args.claude_timeout,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"  Generation failed: {exc}", file=sys.stderr)
            continue

        for candidate in candidates:
            text = candidate["text"]
            if text in seen:
                print(f"  Skipping duplicate: {text!r}")
                continue
            seen.add(text)
            try:
                result = classify(base_url, text, session)
            except (requests.RequestException, ValueError) as exc:
                print(f"ERROR: classification failed for {text!r}: {exc}", file=sys.stderr)
                return 2

            actual = result["decision"]
            expected = candidate["expected"]
            print(f"  [{expected} -> {actual}] {text}")
            if actual != expected:
                print_success(round_number, candidate, result)
                try:
                    add_regression_test(DEFAULT_TEST_PATH, candidate, result)
                except (OSError, RuntimeError, SyntaxError) as exc:
                    print(f"ERROR: could not update tests: {exc}", file=sys.stderr)
                    return 2
                return 0

            attempts.append(
                {
                    "text": text,
                    "expected": expected,
                    "actual": actual,
                    "reason": str(result.get("reason", "")),
                }
            )

    print(
        f"\nNo classifier failure found after {args.max_rounds} rounds "
        f"and {len(seen)} unique candidates.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
