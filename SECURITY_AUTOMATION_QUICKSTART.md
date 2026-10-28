# Quick Start Guide: Security Automation

This guide shows you how to use the security automation system in under 5 minutes.

## For First-Time Users

### Step 1: Run Your First Security Scan

1. Go to the **Actions** tab in GitHub
2. Click on **"Security Audit & Vulnerability Reporting"** in the left sidebar
3. Click **"Run workflow"** button (top right)
4. Keep the default settings:
   - Severity filter: `medium`
   - Create issues: `true`
5. Click the green **"Run workflow"** button

⏱️ The scan takes 2-3 minutes to complete.

### Step 2: Review the Results

Once the workflow completes:

1. **Check the Summary**: Click on the completed workflow run to see:
   - Total findings count
   - Breakdown by severity (Critical, High, Medium, Low)
   - Breakdown by type (Dependency, Code)

2. **Review Created Issues**: Go to the **Issues** tab and filter by `security` label

3. **Download Detailed Reports**: Scroll to the bottom of the workflow run and download the `security-reports` artifact for full JSON reports

### Step 3: Triage and Fix

For each security issue:

1. **Read the description** and understand the vulnerability
2. **Check severity** to prioritize (Critical/High = do first)
3. **Follow the recommended fix** provided in the issue
4. **Test your fix** using the testing instructions
5. **Close the issue** when fixed and deployed

## Example Workflow Run

Here's what you'll see:

```
✅ Checkout repository
✅ Set up Python
✅ Install dependencies
✅ Run Safety check        → Found 2 dependency vulnerabilities
✅ Run pip-audit          → Found 1 additional vulnerability
✅ Run Bandit             → Found 3 code security issues
✅ Run Semgrep            → Found 5 code patterns
✅ Parse results          → Total: 11 findings
✅ Create GitHub Issues   → Created 8 issues (3 duplicates skipped)
```

## Example Issue Output

```markdown
[Security] Dependency: requests vulnerable to CVE-2024-35195

## 🔒 Security Issue

### Severity
**HIGH**

### Category
Dependency Security Issue

### Description
The requests package contains a vulnerability that could allow
an attacker to perform HTTP request smuggling...

### Affected Package
- Package: requests
- Version: 2.32.0
- CVE: CVE-2024-35195

### Recommended Action
Upgrade to requests version 2.32.2 or higher:

1. Update requirements.txt:
   ```
   requests>=2.32.2
   ```

2. Install the update:
   ```bash
   pip install --upgrade requests
   ```

3. Test your application to ensure compatibility
```

## Common Use Cases

### Use Case 1: Weekly Security Review

**When**: Every Monday morning  
**What**: Review automated findings from the scheduled run  
**Who**: Security team or designated developer

```bash
# View all open security issues
gh issue list --label security

# View by severity
gh issue list --label "security,priority: high"
```

### Use Case 2: Pre-Deployment Check

**When**: Before major releases  
**What**: Run a fresh scan to ensure no new vulnerabilities

```bash
# Trigger manually before deployment
gh workflow run security-audit.yml
```

### Use Case 3: After Dependency Updates

**When**: After updating requirements.txt  
**What**: Verify the updates don't introduce new vulnerabilities

```bash
# Run with high severity filter to focus on critical issues
gh workflow run security-audit.yml -f severity_filter=high
```

### Use Case 4: Security Audit Preparation

**When**: Preparing for compliance audits  
**What**: Generate comprehensive security report

```bash
# Run scan and download all reports
gh workflow run security-audit.yml
# Wait for completion, then download artifacts
gh run download <run-id>
```

## Tips & Tricks

### 💡 Tip 1: Adjust Noise Level

Too many issues? Increase the severity threshold:

```yaml
# In .github/workflows/security-audit.yml
severity_filter:
  default: 'high'  # Only report high and critical
```

### 💡 Tip 2: Test Without Creating Issues

Want to see what would be reported without creating issues?

1. Go to Actions → Run workflow
2. Set "Create GitHub issues" to `false`
3. Review the summary and artifacts

### 💡 Tip 3: Quick Fix for Dependencies

Most dependency issues have simple fixes:

```bash
# For a specific package
pip install --upgrade package-name

# Update all packages (carefully!)
pip list --outdated
pip install --upgrade package1 package2
```

### 💡 Tip 4: Prioritize by Business Impact

Not all "High" severity issues are equally urgent:

- **Public-facing API endpoints**: Fix immediately
- **Internal tools/scripts**: Can wait for next sprint
- **Development dependencies**: Lower priority than production

### 💡 Tip 5: Batch Similar Fixes

Group related fixes into one PR:

```bash
git checkout -b fix/security-dependency-updates

# Fix multiple dependency issues at once
pip install --upgrade requests fastapi uvicorn

# Test
python -m unittest discover -s tests

# Commit
git commit -m "fix: update dependencies for security vulnerabilities"
```

## Troubleshooting

### Problem: Workflow Doesn't Run

**Check**:
- Is the workflow file in `.github/workflows/`?
- Do you have Actions enabled in repository settings?
- Are there any syntax errors? (YAML is sensitive to indentation)

**Solution**:
```bash
# Validate workflow syntax
gh workflow list
gh workflow view security-audit.yml
```

### Problem: Too Many False Positives

**Check**: Are the findings relevant to your project?

**Solution**:
1. Add exclusions to Bandit config
2. Create `.semgrep.yml` with ignore rules
3. Increase severity threshold to `high`

### Problem: Missing Dependencies in Scan

**Check**: Are all dependencies in `requirements.txt`?

**Solution**:
```bash
# Generate requirements from current environment
pip freeze > requirements-full.txt

# Compare with requirements.txt
diff requirements.txt requirements-full.txt
```

## Integration with Your Workflow

### For Individual Developers

1. **Before starting work**: Check open security issues
2. **During development**: Consider security implications
3. **Before PR**: Ensure no new security issues introduced
4. **After merge**: Monitor for new automated findings

### For Team Leads

1. **Weekly review**: Check new security issues from automation
2. **Sprint planning**: Allocate time for security fixes
3. **Metrics tracking**: Monitor security debt trends
4. **Process improvement**: Update automation based on learnings

### For Security Team

1. **Triage automation findings**: Validate and prioritize
2. **Define custom rules**: Add project-specific patterns
3. **Track metrics**: Monitor MTTR for security issues
4. **Continuous improvement**: Enhance detection capabilities

## Next Steps

Now that you've run your first scan:

1. ✅ **Fix critical and high severity issues** from the scan
2. ✅ **Set up issue notifications** to stay informed
3. ✅ **Schedule time** for regular security review
4. ✅ **Customize the automation** for your team's needs
5. ✅ **Share feedback** on what works and what doesn't

## Getting Help

- **Documentation**: See `SECURITY_AUTOMATION_README.md` for details
- **Issue Template**: See `SECURITY_AUTOMATION_PROMPT.md` for AI analysis
- **Questions**: Create an issue with label `security-automation`
- **Security concerns**: Follow `SECURITY.md` disclosure policy

---

**Ready to make your codebase more secure?** Start with a single workflow run today! 🔒
