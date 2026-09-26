# Bugbot Auto-Fix Setup Guide

This repository is configured with Bugbot auto-fix capabilities to automatically detect and fix issues in pull requests.

## 🚀 Features

- **Automatic Code Review**: Bugbot analyzes every PR for potential issues
- **Test Failure Detection**: Automatically runs tests and reports failures
- **Syntax Checking**: Validates Python syntax across all files
- **Auto-Fix Capability**: Can automatically generate and push fixes for common issues
- **Smart Comments**: Posts detailed analysis reports on PRs

## 📋 How It Works

### Automatic Triggers

Bugbot runs automatically on:

1. **New Pull Requests**: When a PR is opened
2. **PR Updates**: When new commits are pushed to a PR
3. **Manual Invocation**: When you comment `/bugbot fix` on a PR
4. **Review Comments**: When review comments are added

### Analysis Process

1. **Checkout**: Bugbot checks out the PR branch
2. **Test Execution**: Runs the full test suite (`python -m unittest discover -s tests -v`)
3. **Syntax Validation**: Checks Python syntax on all `.py` files
4. **Report Generation**: Creates a comprehensive analysis report
5. **Auto-Fix** (if issues found): Attempts to automatically fix detected issues

## 🔧 Configuration

### GitHub Repository Setup

#### 1. Enable GitHub Actions

Ensure GitHub Actions is enabled in your repository:
- Go to **Settings** → **Actions** → **General**
- Set **Actions permissions** to "Allow all actions and reusable workflows"

#### 2. Required Permissions

The workflows require the following permissions (already configured):
- `contents: write` - To push fixes to PR branches
- `pull-requests: write` - To comment on PRs
- `issues: write` - To update issue/PR status

#### 3. Branch Protection (Optional)

To ensure Bugbot runs before merges:
- Go to **Settings** → **Branches**
- Add a rule for your main branch
- Enable "Require status checks to pass before merging"
- Select "Bugbot Analysis" as a required check

### Cursor Cloud Agent Integration (Advanced)

For full auto-fix capabilities with Cursor Cloud Agents:

#### Step 1: Install Cursor in Your Repository

```bash
# Add Cursor configuration
mkdir -p .cursor
```

#### Step 2: Add Repository Secrets

Go to **Settings** → **Secrets and variables** → **Actions** and add:

- `CURSOR_API_KEY`: Your Cursor API key (get from [Cursor Dashboard](https://cursor.com/settings))

#### Step 3: Configure Auto-Fix Settings

Create `.cursor/bugbot.json`:

```json
{
  "enabled": true,
  "auto_fix": true,
  "auto_commit": true,
  "commit_message_template": "fix: auto-fix issues detected by Bugbot",
  "analysis": {
    "run_tests": true,
    "check_syntax": true,
    "check_imports": true
  },
  "notifications": {
    "comment_on_pr": true,
    "detailed_reports": true
  }
}
```

#### Step 4: Enable Cursor in Your Workflow

The `bugbot-cursor.yml` workflow includes Cursor integration. When properly configured, it will:

1. Detect issues in PRs
2. Trigger a Cursor Cloud Agent
3. Generate fixes automatically
4. Push commits back to the PR branch
5. Post status updates

## 🎯 Usage

### Automatic Analysis

Bugbot analyzes every PR automatically. No action needed!

When issues are detected, Bugbot will:
- Post a comment with detailed analysis
- Show test failures and syntax errors
- Provide recommendations for fixes

### Manual Fix Trigger

To manually trigger auto-fix on a PR, comment:

```
/bugbot fix
```

Bugbot will:
1. Re-analyze the PR
2. Attempt to generate fixes
3. Push commits to the PR branch (if configured)
4. Post status updates

### Understanding Bugbot Comments

Bugbot comments include:

- **Issues Detected**: Summary of problems found
- **Test Failures**: Which tests failed and why
- **Syntax Issues**: Python syntax errors
- **Auto-Fix Status**: Whether fixes were attempted
- **Next Steps**: Recommendations for resolution

## 📊 Workflow Files

### `bugbot-autofix.yml`

Basic Bugbot workflow that:
- Runs tests and linting
- Posts analysis reports
- Tracks PR status

**Triggers**: PR events (opened, synchronize, reopened)

### `bugbot-cursor.yml`

Advanced workflow with Cursor integration:
- Everything in `bugbot-autofix.yml`
- Plus: Cursor Cloud Agent integration
- Plus: Automatic fix generation and commits

**Triggers**: PR events + manual `/bugbot fix` command

## 🛠️ Customization

### Modify Test Commands

Edit the "Run tests" step in `.github/workflows/bugbot-autofix.yml`:

```yaml
- name: Run tests
  run: |
    # Add your custom test commands here
    python -m pytest tests/ -v
    # or
    python -m unittest discover -s tests -v
```

### Add Custom Checks

Add new steps to the workflow:

```yaml
- name: Custom check
  run: |
    # Your custom validation
    python custom_check.py
```

### Customize Report Format

Modify the "Generate Bugbot report" step to change report formatting.

## 🔍 Monitoring

### View Workflow Runs

1. Go to the **Actions** tab in your repository
2. Select "Bugbot Auto-Fix" or "Bugbot Auto-Fix with Cursor"
3. View individual run details, logs, and artifacts

### Check Status Badges

Add status badges to your README:

```markdown
[![Bugbot](https://github.com/scheduler0/scheduler0-edge-classifier/actions/workflows/bugbot-autofix.yml/badge.svg)](https://github.com/scheduler0/scheduler0-edge-classifier/actions/workflows/bugbot-autofix.yml)
```

## 🐛 Troubleshooting

### Bugbot Not Running

**Check**:
1. GitHub Actions is enabled in repository settings
2. Workflow files are in `.github/workflows/`
3. Required permissions are granted
4. No syntax errors in YAML files

### Auto-Fix Not Working

**Possible causes**:
1. `CURSOR_API_KEY` not configured (for Cursor integration)
2. Branch protection blocking pushes
3. Insufficient permissions in workflow
4. Issues cannot be auto-fixed (require manual intervention)

### Tests Failing in CI But Not Locally

**Check**:
1. All dependencies are listed in `requirements.txt`
2. Python version matches (3.9+ required)
3. External dependencies (e.g., Duckling) are available
4. Environment variables are set correctly

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cursor Documentation](https://cursor.com/docs)
- [Repository Contributing Guide](./CONTRIBUTING.md)
- [Testing Guide](./README.md#testing)

## 🤝 Contributing

To improve Bugbot configuration:

1. Test changes locally first
2. Update this documentation
3. Open a PR with your improvements
4. Ensure all checks pass

## 📝 Example PR Flow

1. **Developer opens PR** → Bugbot analyzes automatically
2. **Issues detected** → Bugbot posts detailed comment
3. **Developer comments `/bugbot fix`** → Auto-fix triggered
4. **Fixes pushed** → Bugbot updates status
5. **All checks pass** → PR ready for review

## 🎉 Success Indicators

When properly configured, you should see:

- ✅ Bugbot comments on every PR
- ✅ Test results reported automatically
- ✅ Auto-fix commits when issues are detected
- ✅ Status checks in PR status section
- ✅ Reduced manual review time

---

**Questions?** Open an issue or check the [Cursor Community](https://cursor.com/community).
