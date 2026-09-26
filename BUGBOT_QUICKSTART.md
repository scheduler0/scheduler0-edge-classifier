# Bugbot Quick Start Guide

Welcome! This guide will help you get started with Bugbot in under 5 minutes.

## ⚡ What is Bugbot?

Bugbot is your automated code quality assistant that:
- Reviews every pull request automatically
- Runs tests and reports failures
- Can fix issues automatically
- Saves review time for your team

## 🚀 For Pull Request Authors

### Creating a PR

When you create a PR, Bugbot automatically:
1. ✅ Runs all tests
2. ✅ Checks code syntax
3. ✅ Posts a detailed analysis report
4. ✅ Updates PR status

**You don't need to do anything - it just works!**

### If Bugbot Finds Issues

Bugbot will post a comment like this:

```
🐛 Bugbot Analysis Report

## Issues Detected

### ❌ Test Failures
- test_intent.py::TestIntentClassifier::test_basic_request - FAILED
- 2 more tests failed

### ⚠️ Recommendations
1. Review the test output above
2. Fix failing tests
3. Push your changes
```

#### Option 1: Fix Manually

Just push your fixes - Bugbot will automatically re-analyze:

```bash
# Fix the code
vim intent.py

# Run tests locally
python -m unittest discover -s tests -v

# Push fixes
git add .
git commit -m "fix: address test failures"
git push
```

#### Option 2: Auto-Fix (if configured)

Comment on your PR:

```
/bugbot fix
```

Bugbot will attempt to fix the issues automatically!

## 🔍 For Code Reviewers

### Understanding Bugbot Reports

Each Bugbot report includes:

1. **Summary**: Quick overview of all checks
2. **Test Results**: Which tests passed/failed
3. **Issues Found**: Specific problems detected
4. **Recommendations**: Actionable next steps

### Using Bugbot During Review

- ✅ **Green checkmark**: All checks passed, ready for detailed review
- ⚠️ **Warning**: Issues found, author should address first
- 🔧 **Auto-fix available**: Can be fixed automatically

### Requesting Specific Checks

You can request Bugbot to analyze specific aspects:

```
@bugbot check error handling
```

(Note: Advanced features require Cursor integration)

## 🎯 Common Scenarios

### Scenario 1: All Tests Pass

```
🐛 Bugbot Analysis Report

## ✅ All Checks Passed

- Tests: ✅ Passing
- Syntax: ✅ Valid

Great work! 🎉
```

**Action**: Proceed with review!

### Scenario 2: Test Failures

```
🐛 Bugbot Analysis Report

## Issues Detected

### ❌ Test Failures
<details>
...test output...
</details>

## 🔧 Auto-Fix Available
```

**Actions**:
1. Review test output
2. Either fix manually or comment `/bugbot fix`
3. Wait for re-analysis

### Scenario 3: Syntax Errors

```
🐛 Bugbot Analysis Report

## Issues Detected

### ⚠️ Syntax Issues
- intent.py: line 42: invalid syntax
```

**Action**: Fix syntax errors (usually quick fixes)

## 📚 Useful Commands

### For PR Authors

| Command | Description |
|---------|-------------|
| `/bugbot fix` | Trigger automatic fix attempt |
| Push new commits | Automatically triggers re-analysis |

### For Reviewers

| Action | Result |
|--------|--------|
| Request changes | Author can use `/bugbot fix` |
| Approve | Bugbot status is part of merge checks |

## 🛠️ Testing Locally

Before pushing, test with Bugbot locally:

```bash
# Run the Bugbot test script
./test-bugbot.sh

# This runs the same checks Bugbot will run
```

## 💡 Pro Tips

1. **Run tests locally first**: Faster feedback than waiting for CI
2. **Check Bugbot status before requesting review**: Save reviewer time
3. **Use `/bugbot fix` for quick fixes**: Let automation help you
4. **Read the full report**: Bugbot provides detailed context

## 🤔 FAQ

### Q: Do I need to configure anything?

**A**: No! Bugbot works automatically on all PRs.

### Q: What if Bugbot is wrong?

**A**: Bugbot is a tool to help, not replace judgment. If you disagree with a suggestion, discuss it in the PR comments.

### Q: Can I disable Bugbot on a specific PR?

**A**: Add `[skip bugbot]` to your PR title. (Use sparingly!)

### Q: How do I get auto-fix working?

**A**: See [BUGBOT_SETUP.md](BUGBOT_SETUP.md) for Cursor integration setup.

### Q: Bugbot isn't running - what's wrong?

**A**: Check:
1. GitHub Actions is enabled
2. Workflow files are present in `.github/workflows/`
3. Branch protection isn't blocking it

## 📖 More Information

- **Full Setup Guide**: [BUGBOT_SETUP.md](BUGBOT_SETUP.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Main README**: [README.md](README.md)

## 🎉 That's It!

You're ready to use Bugbot! It will automatically help with all your PRs.

**Remember**: Bugbot is here to help, not judge. Use it as a tool to catch issues early and save time.

---

Questions? Open an issue or ask in the PR comments!
