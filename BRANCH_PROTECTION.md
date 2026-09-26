# Branch Protection Setup Guide

This guide explains how to configure branch protection rules to ensure tests pass before code is merged into the main branch.

## Overview

A CI workflow has been added (`.github/workflows/test.yml`) that automatically runs tests on:
- All pull requests targeting `main`
- All pushes to `main`
- Manual workflow dispatch

## Setting Up Branch Protection

To enforce that tests must pass before merging, follow these steps:

### 1. Navigate to Branch Protection Settings

1. Go to the repository on GitHub: https://github.com/scheduler0/scheduler0-edge-classifier
2. Click on **Settings** (requires admin access)
3. In the left sidebar, click **Branches**
4. Under "Branch protection rules", click **Add rule** or edit the existing rule for `main`

### 2. Configure Protection Rules

Set the following options:

#### Required Settings

- **Branch name pattern**: `main`
- ✅ **Require a pull request before merging**
  - ✅ **Require approvals**: 1 (recommended)
  - ✅ **Dismiss stale pull request approvals when new commits are pushed** (recommended)
- ✅ **Require status checks to pass before merging**
  - ✅ **Require branches to be up to date before merging**
  - Under "Status checks that are required", add:
    - `test` (this is the name of the job in `.github/workflows/test.yml`)
    - Or search for "Run Unit Tests" once the workflow has run at least once

#### Recommended Additional Settings

- ✅ **Require conversation resolution before merging** - Ensures all PR comments are resolved
- ✅ **Do not allow bypassing the above settings** - Applies rules to administrators too
- ✅ **Restrict who can push to matching branches** - Only allow specific users/teams (optional)

### 3. Save Changes

Click **Create** (or **Save changes** if editing) at the bottom of the page.

## What This Protects Against

With these settings enabled:

1. ❌ **Cannot merge PRs with failing tests** - The "Merge" button will be disabled if tests fail
2. ❌ **Cannot push directly to main** - All changes must go through a PR
3. ✅ **Automated testing** - Every PR automatically runs the test suite
4. ✅ **Visual feedback** - PR pages show test status clearly

## Testing the Setup

1. Create a test branch: `git checkout -b test-ci-setup`
2. Make a small change to a test file
3. Push and create a PR
4. Verify that the "Run Tests" workflow runs automatically
5. Check that you cannot merge until tests pass (if they fail)

## Manual Test Execution

You can also run tests locally before pushing:

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run tests
python -m unittest discover -s tests -v
```

## Workflow Details

The test workflow (`.github/workflows/test.yml`):
- Uses Python 3.12
- Caches pip dependencies for faster runs
- Installs all requirements including spaCy model
- Runs all tests in the `tests/` directory
- Takes approximately 30-60 seconds to complete

## Troubleshooting

### Tests pass locally but fail in CI

- Check Python version (CI uses 3.12)
- Ensure all dependencies are in `requirements.txt`
- Check for environment-specific issues (file paths, etc.)

### "test" status check not appearing

- The workflow must run at least once before it appears in the list
- Push a PR or trigger the workflow manually via Actions tab
- Wait for the workflow to complete, then refresh branch protection settings

### Cannot enable branch protection

- You need admin access to the repository
- Contact the repository owner or admin to set this up

## Additional Resources

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Repository CI/CD Workflow](.github/workflows/test.yml)
