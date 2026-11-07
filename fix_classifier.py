#!/usr/bin/env python3
"""Ask Claude for an intent.py patch that fixes failing classifier tests."""

import argparse
import ast
import difflib
import json
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_DIR = Path(__file__).parent
INTENT_PATH = PROJECT_DIR / "intent.py"
TEST_PATH = PROJECT_DIR / "tests/test_intent.py"
DEFAULT_CLAUDE_PATH = Path.home() / ".local/bin/claude"
CLAUDE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "updated_source": {
            "type": "string",
            "description": "The complete updated contents of intent.py",
        }
    },
    "required": ["updated_source"],
    "additionalProperties": False,
}
TEST_COMMAND = [
    str(PROJECT_DIR / ".venv/bin/python"),
    "-m",
    "unittest",
    "discover",
    "-s",
    "tests",
    "-v",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, approve, and apply a Claude patch for failing tests."
    )
    parser.add_argument(
        "--claude-path",
        type=Path,
        default=DEFAULT_CLAUDE_PATH,
        help=f"Claude Code executable (default: {DEFAULT_CLAUDE_PATH})",
    )
    parser.add_argument(
        "--claude-timeout",
        type=int,
        default=180,
        help="seconds allowed for Claude to generate a patch (default: 180)",
    )
    return parser.parse_args()


def run_tests() -> subprocess.CompletedProcess:
    return subprocess.run(
        TEST_COMMAND,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )


def combined_output(result: subprocess.CompletedProcess) -> str:
    return (result.stdout + result.stderr).strip()


def build_prompt(test_output: str) -> str:
    intent_source = INTENT_PATH.read_text(encoding="utf-8")
    test_source = TEST_PATH.read_text(encoding="utf-8")
    return f"""
Fix the failing scheduling intent-classifier tests by making the smallest
general-purpose change to intent.py. Do not edit tests or hard-code a complete
test prompt. Preserve all currently passing behavior.

Return the complete updated contents of intent.py in the structured
`updated_source` field. Do not return a diff.

Current intent.py:
```python
{intent_source}
```

Current tests/test_intent.py:
```python
{test_source}
```

Failing test output:
```text
{test_output}
```
""".strip()


def generate_patch(claude_path: Path, prompt: str, timeout: int) -> str:
    if not claude_path.is_file():
        raise RuntimeError(f"Claude Code CLI not found at {claude_path}")
    try:
        result = subprocess.run(
            [
                str(claude_path),
                "-p",
                prompt,
                "--output-format",
                "json",
                "--json-schema",
                json.dumps(CLAUDE_OUTPUT_SCHEMA),
                "--tools",
                "",
                "--no-session-persistence",
            ],
            cwd=PROJECT_DIR,
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

    try:
        response = json.loads(result.stdout)
        structured = response.get("structured_output", response)
        updated_output = structured["updated_source"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("Claude did not return the requested updated source") from exc
    if not isinstance(updated_output, str):
        raise ValueError("Claude returned a non-string updated source")

    current_source = INTENT_PATH.read_text(encoding="utf-8")
    updated_source = extract_updated_source(updated_output)
    return build_patch(current_source, updated_source)


# The generated file replaces intent.py wholesale. These names are the
# network boundary. A patch may change classification heuristics, but it may
# not replace the Duckling client, add an import, or reach outside the process.
_PROTECTED_FUNCTIONS = (
    "_normalize_address",
    "_address_is_blocked",
    "_resolved_addresses",
    "_canonical_netloc",
    "_duckling_url_parts",
    "_allowed_addresses",
    "validated_duckling_url",
    "_address_literal",
    "pinned_duckling_url",
    "_close_quietly",
    "_consume_limited",
    "post_duckling",
    "read_duckling_json",
)
_FROZEN_ASSIGNMENTS = (
    "DUCKLING_URL",
    "_BLOCKED_DUCKLING_HOSTS",
    "_AWS_IPV6_IMDS",
    "MAX_DUCKLING_RESPONSE_BYTES",
)
_BANNED_NAMES = {
    "os",
    "socket",
    "requests",
    "subprocess",
    "open",
    "eval",
    "exec",
    "execfile",
    "__import__",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
    "memoryview",
    "__build_class__",
    "__builtins__",
    "builtins",
    "ctypes",
    "importlib",
    "pathlib",
    "pickle",
    "shelve",
    "marshal",
}
_BANNED_ATTRS = {
    "system",
    "popen",
    "fork",
    "forkpty",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "posix_spawn",
    "posix_spawnp",
    "kill",
    "killpg",
    "remove",
    "unlink",
    "rmdir",
    "putenv",
    "unsetenv",
    "urlopen",
    "urlretrieve",
    "Popen",
    "load_module",
}


class _UnsafePatch(ValueError):
    pass


def _import_signature(tree: ast.AST) -> List[str]:
    return [
        ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def _function_map(tree: ast.AST) -> dict:
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, []).append(ast.dump(node))
    return found


def _assignment_map(tree: ast.AST) -> dict:
    found: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found.setdefault(target.id, []).append(ast.dump(node))
    return found


def _assigned_names(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: List[str] = []
        for element in node.elts:
            names.extend(_assigned_names(element))
        return names
    if isinstance(node, ast.Starred):
        return _assigned_names(node.value)
    return []


def _iter_untrusted(tree: ast.AST):
    protected = set(_PROTECTED_FUNCTIONS)
    frozen = set(_FROZEN_ASSIGNMENTS)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in protected:
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in frozen for target in node.targets
        ):
            continue
        yield from ast.walk(node)


def assert_patch_source_is_safe(source: str) -> None:
    """Reject a generated intent.py that can leave the Duckling boundary.

    Direct calls such as ``os.system`` were already refused. Indirect forms
    were not: ``requests.Session().post``, ``os.execlp``, ``open``, a lambda
    that reaches ``__import__``, ``os.environ`` reads, and dunder traversal
    (``__class__.__subclasses__``). The network helpers and their module-level
    configuration have to stay identical to the file on disk.
    """
    try:
        updated = ast.parse(source)
        baseline = ast.parse(INTENT_PATH.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise _UnsafePatch("generated patch is not valid Python") from exc

    if _import_signature(updated) != _import_signature(baseline):
        raise _UnsafePatch("generated patch changes imports")

    baseline_functions = _function_map(baseline)
    updated_functions = _function_map(updated)
    for name in _PROTECTED_FUNCTIONS:
        if updated_functions.get(name) != baseline_functions.get(name):
            raise _UnsafePatch(f"generated patch changes {name}")

    baseline_assignments = _assignment_map(baseline)
    updated_assignments = _assignment_map(updated)
    for name in _FROZEN_ASSIGNMENTS:
        if updated_assignments.get(name) != baseline_assignments.get(name):
            raise _UnsafePatch(f"generated patch changes {name}")

    frozen_names = set(_PROTECTED_FUNCTIONS) | set(_FROZEN_ASSIGNMENTS)
    for node in _iter_untrusted(updated):
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise _UnsafePatch(f"generated patch references {node.id}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in _BANNED_ATTRS
        ):
            raise _UnsafePatch(f"generated patch uses .{node.attr}")
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise _UnsafePatch("generated patch uses an indirect call")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in frozen_names or node.name in _BANNED_NAMES:
                raise _UnsafePatch(f"generated patch redefines {node.name}")
        targets: List[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                targets.extend(_assigned_names(target))
        elif isinstance(node, ast.AnnAssign):
            targets.extend(_assigned_names(node.target))
        elif isinstance(node, (ast.AugAssign, ast.NamedExpr)):
            targets.extend(_assigned_names(node.target))
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                targets.extend(_assigned_names(target))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            targets.extend(node.names)
        for name in targets:
            if name in frozen_names:
                raise _UnsafePatch(f"generated patch rebinds {name}")


def extract_updated_source(output: str) -> str:
    text = output.strip()
    opening_fence = text.find("```")
    if opening_fence != -1:
        first_newline = text.find("\n", opening_fence)
        closing_fence = text.rfind("\n```")
        if first_newline == -1 or closing_fence <= first_newline:
            raise ValueError("Claude returned an incomplete code fence")
        text = text[first_newline + 1 : closing_fence]

    updated_source = text.rstrip() + "\n"
    try:
        compile(updated_source, str(INTENT_PATH), "exec")
    except SyntaxError as exc:
        first_line = updated_source.splitlines()[0] if updated_source else "(empty)"
        raise ValueError(
            f"Claude returned invalid Python: {exc}; first line was {first_line!r}"
        ) from exc
    if "def classify(" not in updated_source:
        raise ValueError("Claude response does not appear to contain intent.py")
    assert_patch_source_is_safe(updated_source)
    return updated_source


def build_patch(current_source: str, updated_source: str) -> str:
    if current_source == updated_source:
        raise ValueError("Claude did not change intent.py")
    unified_diff = "".join(
        difflib.unified_diff(
            current_source.splitlines(keepends=True),
            updated_source.splitlines(keepends=True),
            fromfile="a/intent.py",
            tofile="b/intent.py",
        )
    )
    patch = "diff --git a/intent.py b/intent.py\n" + unified_diff
    validate_patch_paths(patch)
    return patch


def validate_patch_paths(patch: str) -> None:
    diff_headers: List[str] = [
        line for line in patch.splitlines() if line.startswith("diff --git ")
    ]
    if not diff_headers:
        raise ValueError("patch has no file diff")
    if any(header != "diff --git a/intent.py b/intent.py" for header in diff_headers):
        raise ValueError("patch may only modify intent.py")

    for line in patch.splitlines():
        if line.startswith("--- ") and line != "--- a/intent.py":
            raise ValueError("patch contains an unexpected source path")
        if line.startswith("+++ ") and line != "+++ b/intent.py":
            raise ValueError("patch contains an unexpected destination path")


def confirm(message: str) -> bool:
    try:
        answer = input(f"{message} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def git_apply(patch: str, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "apply", *arguments, "-"],
        cwd=PROJECT_DIR,
        input=patch,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = parse_args()
    initial_test = run_tests()
    if initial_test.returncode == 0:
        print(combined_output(initial_test))
        print("\nAll classifier tests already pass; no patch is needed.")
        return 0

    test_output = combined_output(initial_test)
    print(test_output)
    print("\nAsking Claude to generate a minimal intent.py patch...")
    try:
        patch = generate_patch(
            args.claude_path.expanduser(),
            build_prompt(test_output),
            args.claude_timeout,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    check = git_apply(patch, "--check")
    if check.returncode != 0:
        print(f"ERROR: Claude patch does not apply cleanly:\n{check.stderr}", file=sys.stderr)
        return 2

    print("\nProposed patch:\n")
    print(patch)
    if not confirm("Apply this patch to intent.py and run the tests?"):
        print("Patch was not applied.")
        return 1

    applied = git_apply(patch)
    if applied.returncode != 0:
        print(f"ERROR: failed to apply patch:\n{applied.stderr}", file=sys.stderr)
        return 2

    final_test = run_tests()
    print("\nTest result after applying patch:\n")
    print(combined_output(final_test))
    if final_test.returncode == 0:
        print("\nSUCCESS: patch applied and all classifier tests pass.")
        return 0

    reverted = git_apply(patch, "--reverse")
    if reverted.returncode == 0:
        print(
            "\nERROR: tests still failed; the generated patch was reverted.",
            file=sys.stderr,
        )
    else:
        print(
            "\nERROR: tests failed and the patch could not be reverted automatically. "
            "Inspect intent.py before continuing.",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
