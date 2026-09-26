#!/bin/bash
# test-bugbot.sh - Script to test Bugbot configuration locally

set -e

echo "🧪 Testing Bugbot Configuration"
echo "================================"
echo

# Check if running in GitHub Actions
if [ -n "$GITHUB_ACTIONS" ]; then
    echo "✅ Running in GitHub Actions environment"
else
    echo "ℹ️  Running locally (not in GitHub Actions)"
fi

echo

# Check Python version
echo "1️⃣ Checking Python version..."
python --version || python3 --version
echo "   ✅ Python available"
echo

# Check if requirements.txt exists
echo "2️⃣ Checking dependencies file..."
if [ -f "requirements.txt" ]; then
    echo "   ✅ requirements.txt found"
    echo "   Installing dependencies..."
    pip install -q -r requirements.txt
    python -m spacy download en_core_web_sm --quiet
    echo "   ✅ Dependencies installed"
else
    echo "   ❌ requirements.txt not found"
    exit 1
fi
echo

# Check if tests directory exists
echo "3️⃣ Checking tests directory..."
if [ -d "tests" ]; then
    echo "   ✅ tests/ directory found"
    TEST_FILES=$(find tests -name "test_*.py" | wc -l)
    echo "   Found $TEST_FILES test file(s)"
else
    echo "   ❌ tests/ directory not found"
    exit 1
fi
echo

# Run tests
echo "4️⃣ Running tests..."
set +e
python -m unittest discover -s tests -v 2>&1 | tee test_output.txt
TEST_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "   ✅ All tests passed!"
else
    echo "   ⚠️  Some tests failed (exit code: $TEST_EXIT_CODE)"
    echo
    echo "   Failed test summary:"
    grep -E "(FAIL|ERROR)" test_output.txt || echo "   (See full output above)"
fi
echo

# Check Python syntax
echo "5️⃣ Checking Python syntax..."
set +e
SYNTAX_ERRORS=0
for file in $(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*"); do
    python -m py_compile "$file" 2>&1
    if [ $? -ne 0 ]; then
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
    fi
done
set -e

if [ $SYNTAX_ERRORS -eq 0 ]; then
    echo "   ✅ No syntax errors found"
else
    echo "   ⚠️  Found $SYNTAX_ERRORS syntax error(s)"
fi
echo

# Check if Bugbot workflows exist
echo "6️⃣ Checking Bugbot workflows..."
WORKFLOW_DIR=".github/workflows"
if [ -d "$WORKFLOW_DIR" ]; then
    if [ -f "$WORKFLOW_DIR/bugbot-autofix.yml" ]; then
        echo "   ✅ bugbot-autofix.yml found"
    else
        echo "   ❌ bugbot-autofix.yml not found"
    fi
    
    if [ -f "$WORKFLOW_DIR/bugbot-cursor.yml" ]; then
        echo "   ✅ bugbot-cursor.yml found"
    else
        echo "   ❌ bugbot-cursor.yml not found"
    fi
else
    echo "   ❌ .github/workflows/ directory not found"
fi
echo

# Check Cursor configuration
echo "7️⃣ Checking Cursor configuration..."
if [ -f ".cursor/bugbot.json" ]; then
    echo "   ✅ .cursor/bugbot.json found"
    echo "   Configuration preview:"
    cat .cursor/bugbot.json | head -10
    echo "   ..."
else
    echo "   ⚠️  .cursor/bugbot.json not found (optional)"
fi
echo

# Final summary
echo "================================"
echo "📊 Summary"
echo "================================"
echo

if [ $TEST_EXIT_CODE -eq 0 ] && [ $SYNTAX_ERRORS -eq 0 ]; then
    echo "✅ All checks passed! Bugbot is ready."
    echo
    echo "Next steps:"
    echo "  1. Commit and push these changes"
    echo "  2. Open a pull request"
    echo "  3. Watch Bugbot analyze your PR automatically"
    echo
    exit 0
else
    echo "⚠️  Some issues detected:"
    [ $TEST_EXIT_CODE -ne 0 ] && echo "  - Test failures"
    [ $SYNTAX_ERRORS -ne 0 ] && echo "  - Syntax errors"
    echo
    echo "These issues will be detected by Bugbot when you open a PR."
    echo "Bugbot can help fix them automatically!"
    echo
    exit 1
fi
