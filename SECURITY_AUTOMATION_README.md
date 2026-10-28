# Security Automation System

This repository includes an automated security analysis and vulnerability reporting system that helps maintain the security posture of the scheduler0-edge-classifier project.

## Overview

The security automation system consists of:

1. **Detailed Analysis Guide** (`SECURITY_AUTOMATION_PROMPT.md`) - Comprehensive prompt for AI-powered security analysis
2. **GitHub Actions Workflow** (`.github/workflows/security-audit.yml`) - Automated security scanning pipeline
3. **Issue Templates** - Structured format for security vulnerability reporting

## How It Works

### Automated Weekly Scans

The system runs automatically every Monday at 9 AM UTC and performs:

- **Dependency Vulnerability Scanning** - Checks Python packages for known CVEs
- **Code Security Analysis** - Identifies common security issues and anti-patterns
- **Container Security Review** - Analyzes Docker configuration for security issues
- **Infrastructure Security Check** - Reviews deployment configurations

### Security Tools Used

| Tool | Purpose | What It Checks |
|------|---------|----------------|
| **Safety** | Python dependency vulnerabilities | Known CVEs in requirements.txt |
| **pip-audit** | PyPI package security | Security advisories from PyPI |
| **Bandit** | Python code security | Common security anti-patterns |
| **Semgrep** | Advanced code patterns | Complex security vulnerabilities |

## Manual Triggering

You can manually trigger a security audit:

1. Go to **Actions** → **Security Audit & Vulnerability Reporting**
2. Click **Run workflow**
3. Configure options:
   - **Severity filter**: Choose minimum severity level (critical, high, medium, low)
   - **Create issues**: Enable/disable automatic issue creation

## What Gets Created

When vulnerabilities are found, the system automatically creates GitHub issues with:

- **Severity Rating** (Critical, High, Medium, Low)
- **Detailed Description** of the vulnerability
- **Security Impact Assessment** (confidentiality, integrity, availability)
- **Recommended Fix** with code examples
- **Testing Instructions** to verify the fix
- **References** to CVE databases and security advisories

### Example Issue Structure

```markdown
[Security] Dependency: FastAPI vulnerable to CVE-2024-XXXXX

## 🔒 Security Issue

### Severity
**HIGH**

### Category
Dependency Security Issue

### Description
FastAPI version X.Y.Z contains a known vulnerability...

### Affected Package
- Package: fastapi
- Version: 0.111.0
- CVE: CVE-2024-XXXXX

### Recommended Action
Upgrade to FastAPI version X.Y.Z or higher...
```

## Using the AI-Powered Analysis

For more comprehensive analysis, use the `SECURITY_AUTOMATION_PROMPT.md` with:

### Cursor Cloud Agent

1. Create a new Cloud Agent
2. Copy the content of `SECURITY_AUTOMATION_PROMPT.md`
3. Run: "Analyze this repository using the security automation prompt"

### GitHub Automation

Create a GitHub automation that:
1. Runs on a schedule (weekly/monthly)
2. Uses the prompt to guide analysis
3. Creates issues automatically for findings

### Manual Review

Use the prompt as a checklist when:
- Conducting security audits
- Reviewing pull requests for security issues
- Preparing for compliance audits
- Onboarding new security team members

## Security Finding Categories

The system checks for:

### 1. Dependency Vulnerabilities ⚠️
- Known CVEs in Python packages
- Outdated dependencies with security patches
- Supply chain security risks

### 2. Code Security Issues 🔒
- Input validation problems
- Injection vulnerabilities (command, code)
- Error handling and information disclosure
- Authentication and authorization gaps

### 3. Infrastructure Security 🏗️
- Docker container security (root user, image tags)
- AWS security (IAM, security groups, VPC)
- Secrets management issues
- GitHub Actions workflow security

### 4. Data Privacy 🛡️
- PII handling in logs
- Data retention policies
- Compliance considerations (GDPR, etc.)

### 5. API Security 🌐
- Missing rate limiting
- Lack of authentication
- DoS vulnerabilities
- CORS misconfigurations

## Reviewing Security Issues

When a security issue is created:

1. **Triage** - Review severity and impact
2. **Validate** - Confirm the vulnerability exists
3. **Plan** - Prioritize based on severity
4. **Fix** - Implement recommended fixes
5. **Test** - Verify the fix works
6. **Deploy** - Roll out to production
7. **Close** - Close the issue with notes

## Severity Guidelines

| Severity | Description | Response Time |
|----------|-------------|---------------|
| 🔴 **Critical** | Exploitable vulnerabilities with high impact | Immediate (< 24 hours) |
| 🟠 **High** | Serious security issues requiring prompt attention | 1-3 days |
| 🟡 **Medium** | Security improvements that should be addressed | 1-2 weeks |
| 🟢 **Low** | Minor issues or defense-in-depth enhancements | As time permits |

## Integration with Existing Systems

### Works with Bugbot

The security automation complements the existing Bugbot system:
- **Bugbot**: Catches bugs and code quality issues in PRs
- **Security Automation**: Proactive vulnerability scanning

### Integrates with CI/CD

The security workflow runs alongside existing workflows:
- `test.yml` - Unit and integration tests
- `deploy.yml` - Deployment pipeline
- `security-audit.yml` - **NEW** Security scanning

## Customization

### Adjusting Scan Frequency

Edit `.github/workflows/security-audit.yml`:

```yaml
on:
  schedule:
    # Run daily instead of weekly
    - cron: '0 9 * * *'
```

### Changing Severity Threshold

Modify the workflow default:

```yaml
severity_filter:
  default: 'high'  # Only report high and critical
```

### Adding Custom Checks

Add new scanning tools in the workflow:

```yaml
- name: Run custom security tool
  run: |
    custom-tool --scan . --output report.json
```

## Monitoring and Metrics

Track security posture over time:

- **Open Security Issues** - Number of unresolved findings
- **Mean Time to Resolution** - Average time to fix issues by severity
- **Vulnerability Trends** - Are new issues appearing faster than fixes?
- **Coverage** - Are all components being scanned?

View metrics in:
- GitHub Issues dashboard with `security` label
- GitHub Actions workflow history
- Security tab (for GitHub Advanced Security users)

## Best Practices

### For Developers

- ✅ Review security issues assigned to you promptly
- ✅ Test fixes thoroughly before deploying
- ✅ Update the issue with your findings and fix details
- ✅ Consider security in code reviews

### For Security Team

- ✅ Review automated findings weekly
- ✅ Validate before assigning to developers
- ✅ Track resolution metrics
- ✅ Update the automation prompt based on learnings

### For Repository Admins

- ✅ Ensure GitHub secrets are properly configured
- ✅ Monitor workflow execution for failures
- ✅ Keep security scanning tools updated
- ✅ Review and approve security-related PRs quickly

## Troubleshooting

### Workflow Fails

**Issue**: Security audit workflow fails to complete

**Solution**:
1. Check workflow logs in Actions tab
2. Ensure scanning tools are compatible with dependencies
3. Verify GitHub token has `issues: write` permission

### Too Many False Positives

**Issue**: Automated scan creates too many invalid issues

**Solution**:
1. Adjust severity threshold to `high` or `critical`
2. Add false positive filtering in the Python script
3. Improve de-duplication logic

### Missing Vulnerabilities

**Issue**: Known issues not being detected

**Solution**:
1. Update security scanning tools to latest versions
2. Add custom Semgrep rules for project-specific patterns
3. Supplement with manual code review

## Contributing

To improve the security automation:

1. Add new security checks to the prompt
2. Enhance the GitHub workflow with additional tools
3. Improve issue templates based on feedback
4. Share learnings with the team

## Resources

### Internal Documentation
- [SECURITY.md](./SECURITY.md) - Security policy and vulnerability reporting
- [SECURITY_AUTOMATION_PROMPT.md](./SECURITY_AUTOMATION_PROMPT.md) - Detailed analysis guide
- [CONTRIBUTING.md](./CONTRIBUTING.md) - Contribution guidelines

### External Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [GitHub Security Best Practices](https://docs.github.com/en/code-security)

## Support

For questions or issues with the security automation:

1. Check the troubleshooting section above
2. Review workflow logs in the Actions tab
3. Create an issue with the `security-automation` label
4. Contact the security team

---

**Remember**: Security is everyone's responsibility. This automation helps, but human review and judgment are essential.
