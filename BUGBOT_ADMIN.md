# Bugbot Administration Guide

This guide is for repository administrators and team leads setting up and managing Bugbot.

## 📋 Table of Contents

- [Overview](#overview)
- [Initial Setup](#initial-setup)
- [Configuration Management](#configuration-management)
- [Access Control](#access-control)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

### What Administrators Need to Know

Bugbot consists of three integration levels:

1. **Level 1 (Basic)**: Works automatically, no setup needed
2. **Level 2 (Manual Trigger)**: Works automatically, users can trigger fixes
3. **Level 3 (Full Auto-Fix)**: Requires Cursor API configuration

### Architecture

```
Pull Request Event
       ↓
GitHub Actions Trigger
       ↓
Bugbot Workflow
       ↓
   ┌────────────────┐
   │ Run Tests      │
   │ Check Syntax   │
   │ Analyze Issues │
   └────────────────┘
       ↓
   Post Analysis Comment
       ↓
   [Optional: Auto-Fix]
       ↓
   Update PR Status
```

## Initial Setup

### Step 1: Verify Workflows are Active

After merging the Bugbot PR, verify the workflows are enabled:

1. Go to **Settings** → **Actions** → **General**
2. Ensure "Allow all actions and reusable workflows" is selected
3. Check that workflows appear in the **Actions** tab

### Step 2: Configure Permissions

Ensure the workflows have necessary permissions:

**Repository Settings** → **Actions** → **General** → **Workflow permissions**:
- ✅ Read and write permissions
- ✅ Allow GitHub Actions to create and approve pull requests

### Step 3: Test Basic Functionality

Create a test PR to verify Bugbot runs:

```bash
# Create a test branch
git checkout -b test-bugbot-setup

# Make a trivial change
echo "# Test" >> README.md
git add README.md
git commit -m "test: verify Bugbot runs"
git push -u origin test-bugbot-setup

# Open PR via GitHub UI or gh CLI
gh pr create --title "test: Bugbot verification" --body "Testing Bugbot setup"
```

Verify:
- ✅ Bugbot workflow runs automatically
- ✅ Analysis comment appears on PR
- ✅ Status check shows in PR status section

### Step 4: Configure Branch Protection (Optional)

To require Bugbot checks before merging:

1. Go to **Settings** → **Branches**
2. Add or edit rule for `main` (or your default branch)
3. Enable: **Require status checks to pass before merging**
4. Select: **Bugbot Analysis** and **Bugbot Auto-Fix**
5. Save changes

## Configuration Management

### Workflow Files

Located in `.github/workflows/`:

| File | Purpose | Can Modify? |
|------|---------|-------------|
| `bugbot-autofix.yml` | Basic analysis | Yes, carefully |
| `bugbot-cursor.yml` | Cursor integration | Yes, carefully |
| `bugbot-cloud-agent.yml` | Manual dispatch | Yes |

### Configuration File

`.cursor/bugbot.json` contains runtime configuration:

```json
{
  "bugbot": {
    "enabled": true,
    "auto_fix": true,
    "auto_commit": true,
    "analysis": {
      "run_tests": true,
      "test_command": "python -m unittest discover -s tests -v",
      "check_syntax": true
    }
  }
}
```

**Safe to modify**:
- `test_command` - Adjust for your test framework
- `commit_message_template` - Customize commit messages
- `notifications` settings - Control report format

**Be careful with**:
- `enabled` - Disables Bugbot entirely
- `auto_fix` - Controls auto-fix behavior
- `auto_commit` - Controls automatic commits

### Customizing for Your Project

#### Custom Test Command

Edit `.cursor/bugbot.json`:

```json
{
  "bugbot": {
    "analysis": {
      "test_command": "pytest tests/ -v --cov"
    }
  }
}
```

Then update workflows to use this configuration.

#### Additional Checks

Add steps to `bugbot-autofix.yml`:

```yaml
- name: Run linting
  run: |
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

- name: Check formatting
  run: |
    black --check .
```

#### Custom Notifications

Modify the "Generate Bugbot report" step to customize output format.

## Access Control

### Who Can Trigger Manual Fixes?

By default, anyone who can comment on a PR can trigger `/bugbot fix`.

To restrict:

1. Add a check in `bugbot-cursor.yml`:

```yaml
- name: Check permissions
  if: github.event_name == 'issue_comment'
  uses: actions/github-script@v7
  with:
    script: |
      const allowedUsers = ['username1', 'username2'];
      const actor = context.actor;
      
      if (!allowedUsers.includes(actor)) {
        core.setFailed('Only maintainers can trigger auto-fix');
      }
```

### Repository Secrets

For Level 3 (Full Auto-Fix), configure secrets:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add **New repository secret**:
   - Name: `CURSOR_API_KEY`
   - Value: [Your Cursor API key from cursor.com]

**Security Note**: Never commit API keys to the repository.

### Team Access

Configure via `.github/CODEOWNERS`:

```
# Bugbot configuration
/.github/workflows/bugbot*.yml @org/admin-team
/.cursor/ @org/admin-team
```

## Monitoring and Maintenance

### Dashboard

Monitor Bugbot activity:

1. **Actions Tab**: View all workflow runs
2. **Insights** → **Dependency graph** → **Dependencies**: Check security
3. **Settings** → **Actions** → **Actions usage**: Monitor quota

### Key Metrics to Track

| Metric | What to Look For |
|--------|------------------|
| Success Rate | Should be >95% |
| Average Run Time | Should be <5 minutes |
| Auto-Fix Attempts | Track effectiveness |
| False Positives | Issues incorrectly flagged |

### Regular Maintenance Tasks

#### Weekly

- [ ] Review failed workflow runs
- [ ] Check for false positives
- [ ] Monitor GitHub Actions quota usage

#### Monthly

- [ ] Review Bugbot configuration
- [ ] Update documentation if behavior changed
- [ ] Check for workflow/action updates
- [ ] Review team feedback

#### Quarterly

- [ ] Audit effectiveness (issues caught vs. missed)
- [ ] Survey team satisfaction
- [ ] Consider advanced features
- [ ] Review access controls

### Logs and Artifacts

Bugbot uploads artifacts for each run:

1. Go to **Actions** tab
2. Select a workflow run
3. Scroll to **Artifacts** section
4. Download:
   - `test_output.txt` - Full test results
   - `lint_output.txt` - Linting results
   - `bugbot_report.md` - Generated report

Retention: 7-30 days (configurable)

## Troubleshooting

### Common Issues

#### Bugbot Not Running on PRs

**Symptoms**: No analysis comment, no workflow run

**Diagnose**:
```bash
# Check if workflows are enabled
# GitHub UI: Settings → Actions → General

# Verify workflow files exist
ls -la .github/workflows/bugbot*.yml

# Check workflow syntax
gh workflow list
```

**Fix**:
1. Enable Actions in repository settings
2. Verify workflow files are committed to main branch
3. Check for syntax errors in YAML files

#### Tests Passing Locally but Failing in Bugbot

**Symptoms**: Tests pass on developer machine but fail in CI

**Diagnose**:
```bash
# Check Python version
# In workflow: uses: actions/setup-python@v5 with python-version: '3.9'

# Check dependencies
git diff main -- requirements.txt

# Review environment differences
cat .github/workflows/bugbot-autofix.yml | grep -A 10 "Install dependencies"
```

**Fix**:
1. Match Python version in workflow to development
2. Ensure all dependencies in `requirements.txt`
3. Add missing system dependencies to workflow

#### Auto-Fix Not Working

**Symptoms**: `/bugbot fix` comment doesn't trigger fixes

**Diagnose**:
```bash
# Check if workflow is triggered
# Actions tab → filter by PR number

# Verify Cursor API key (if Level 3)
# Settings → Secrets → Check CURSOR_API_KEY exists

# Check workflow logs
gh run list --workflow=bugbot-cursor.yml
gh run view <run-id> --log
```

**Fix**:
1. Verify workflow permissions (write access to contents)
2. Check Cursor API key is configured
3. Review workflow logs for errors
4. Ensure branch is not protected from force pushes

#### Excessive False Positives

**Symptoms**: Bugbot flags issues that aren't real problems

**Diagnose**:
- Review recent Bugbot comments
- Identify patterns in false positives
- Check test suite reliability

**Fix**:
1. Adjust test reliability:
   ```python
   # Add retries for flaky tests
   @retry(tries=3, delay=1)
   def test_flaky_feature(self):
       ...
   ```

2. Update Bugbot configuration:
   ```json
   {
     "analysis": {
       "check_syntax": true,
       "check_imports": false  // Disable if too strict
     }
   }
   ```

3. Add exclusions to workflow:
   ```yaml
   - name: Run tests
     run: |
       python -m unittest discover -s tests -v -k 'not flaky'
   ```

### Getting Help

1. **Check logs**: Actions tab → Select run → View logs
2. **Review documentation**: BUGBOT_SETUP.md, BUGBOT_QUICKSTART.md
3. **Open issue**: Use "Bugbot" label
4. **Team discussion**: Discuss in team chat/standup

## Advanced Topics

### Integrating with Other CI Systems

Bugbot can work alongside other CI/CD:

```yaml
# In .github/workflows/main.yml
jobs:
  integration:
    runs-on: ubuntu-latest
    needs: bugbot-analyze  # Wait for Bugbot
    steps:
      - name: Deploy if Bugbot passed
        run: ./deploy.sh
```

### Custom Fix Strategies

For Level 3, customize fix strategies in `.cursor/bugbot.json`:

```json
{
  "fix_strategies": {
    "test_failures": "analyze_and_fix",
    "syntax_errors": "auto_fix",
    "import_errors": "suggest_only"
  }
}
```

### Scheduled Analysis

Run Bugbot on a schedule to catch regressions:

```yaml
# Add to bugbot-autofix.yml
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  pull_request:
    types: [opened, synchronize]
```

### Multi-Repository Setup

For organizations with multiple repos:

1. Create a template repository with Bugbot configuration
2. Use repository templates for new projects
3. Centralize configuration via organization defaults
4. Use composite actions for shared logic

### Metrics and Reporting

Generate reports on Bugbot effectiveness:

```bash
# Script: generate_bugbot_metrics.sh
#!/bin/bash

# Count PRs with Bugbot comments
TOTAL_PRS=$(gh pr list --state all --json number | jq length)
BUGBOT_COMMENTS=$(gh issue list --state all --search "🐛 Bugbot Analysis" | wc -l)

echo "Total PRs: $TOTAL_PRS"
echo "Bugbot analyzed: $BUGBOT_COMMENTS"
echo "Coverage: $(echo "scale=2; $BUGBOT_COMMENTS / $TOTAL_PRS * 100" | bc)%"
```

### Notification Integration

Send Bugbot alerts to Slack/Teams:

```yaml
- name: Notify Slack
  if: steps.test.outputs.exit_code != '0'
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Bugbot found issues in PR #${{ github.event.pull_request.number }}",
        "url": "${{ github.event.pull_request.html_url }}"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

## Best Practices

### Do's ✅

- **Test changes to workflows in a feature branch first**
- **Document any customizations**
- **Monitor false positive rate**
- **Respond to team feedback**
- **Keep workflows simple and focused**
- **Use semantic versioning for workflow actions**

### Don'ts ❌

- **Don't commit API keys or secrets**
- **Don't make workflows too complex**
- **Don't ignore failed runs**
- **Don't block deployments on flaky tests**
- **Don't modify workflows without testing**

## Support and Resources

- **Documentation**: BUGBOT_SETUP.md, BUGBOT_QUICKSTART.md
- **GitHub Actions**: https://docs.github.com/en/actions
- **Cursor Documentation**: https://cursor.com/docs
- **Team Discussion**: Use your team's communication channel

## Appendix

### Workflow Permissions Reference

```yaml
permissions:
  contents: write       # Push commits, create branches
  pull-requests: write  # Comment on PRs, update status
  issues: write        # Comment on issues/PRs
```

### Useful Commands

```bash
# List workflows
gh workflow list

# View recent runs
gh run list --workflow=bugbot-autofix.yml --limit 10

# View logs for specific run
gh run view <run-id> --log

# Re-run failed workflow
gh run rerun <run-id>

# Trigger manual workflow
gh workflow run bugbot-cloud-agent.yml -f pr_number=123 -f fix_mode=analyze_only
```

---

**Questions or issues?** Open an issue with the "Bugbot" label or contact the DevOps team.
