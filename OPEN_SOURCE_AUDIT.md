# Open Source Readiness Audit

**Repository**: scheduler0-edge-classifier  
**Audit Date**: September 26, 2026  
**Current Status**: Private GitHub Repository  

## Executive Summary

This document outlines security and documentation gaps that should be addressed before making the scheduler0-edge-classifier repository public. The repository contains an NLP-based intent classification service for Scheduler0, deployed on AWS ECS.

---

## 🔴 Critical Security Issues (Must Fix Before Public Release)

### 1. AWS Account ID Exposure
**Severity**: HIGH  
**Files Affected**:
- `production-scheduler0-intent-classifier-ecs-task-definition.json`
- `staging-scheduler0-intent-classifier-ecs-task-definition.json`

**Issue**: Hardcoded AWS account ID `748201447723` appears in ECR image URIs and IAM role ARNs.

**Recommendation**:
```bash
# Option A: Use environment variables in task definitions
"image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/..."

# Option B: Add to .gitignore and provide templates
*.json → *-template.json
```

**Action Items**:
- [ ] Create template files with placeholders (e.g., `<AWS_ACCOUNT_ID>`)
- [ ] Move actual task definitions to `.gitignore`
- [ ] Document how to generate real task definitions from templates
- [ ] Add `.env.example` showing required environment variables

### 2. GitHub Secrets Documentation
**Severity**: MEDIUM  
**Files Affected**: `.github/workflows/_deploy.yml`

**Issue**: Workflow requires AWS credentials via GitHub secrets but doesn't document:
- Required secret names (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Required variables (`AWS_REGION`)
- IAM permissions needed

**Recommendation**: Create a security documentation file explaining secret management.

**Action Items**:
- [ ] Document all required GitHub secrets/variables
- [ ] Specify minimum IAM permissions needed
- [ ] Add setup instructions for CI/CD

### 3. Internal Infrastructure References
**Severity**: MEDIUM  

**Issue**: Repository assumes internal AWS infrastructure exists:
- ECS cluster names (`staging_scheduler0_ecs_cluster`)
- Log groups (`ecs/production-scheduler0-intent-classifier-ecs-service`)
- IAM roles with specific naming conventions
- EC2 instances with custom attributes

**Recommendation**: Add infrastructure-as-code templates or clear documentation.

**Action Items**:
- [ ] Provide Terraform/CloudFormation examples
- [ ] Document infrastructure prerequisites
- [ ] Create deployment architecture diagram

---

## 📄 Documentation Gaps (Should Add Before Public Release)

### 1. Missing LICENSE File
**Severity**: HIGH  
**Current State**: No license file present  

**Recommendation**: Add an appropriate open-source license.

**Popular Options**:
- **MIT License**: Most permissive, good for libraries
- **Apache 2.0**: Includes patent grant, good for commercial use
- **BSD 3-Clause**: Permissive with attribution requirement

**Action Items**:
- [ ] Choose and add LICENSE file
- [ ] Add license badge to README
- [ ] Add copyright headers to source files (optional but recommended)

### 2. Missing CONTRIBUTING.md
**Severity**: MEDIUM  

**Should Include**:
- How to set up development environment
- Code style guidelines
- How to run tests
- Pull request process
- Code of conduct expectations

**Action Items**:
- [ ] Create CONTRIBUTING.md with development workflow
- [ ] Define coding standards (PEP 8 for Python, etc.)
- [ ] Document commit message conventions

### 3. Missing CODE_OF_CONDUCT.md
**Severity**: LOW (but important for community projects)  

**Recommendation**: Add a code of conduct (e.g., Contributor Covenant).

**Action Items**:
- [ ] Add CODE_OF_CONDUCT.md (use standard template)

### 4. Missing SECURITY.md
**Severity**: MEDIUM  

**Should Include**:
- How to report security vulnerabilities
- Response timeline expectations
- Supported versions
- Security update policy

**Action Items**:
- [ ] Create SECURITY.md with vulnerability reporting process
- [ ] Set up private security advisory capability on GitHub

### 5. Incomplete README.md
**Current README Issues**:
- No project description/purpose in first paragraph
- No badges (build status, license, version)
- No "Getting Started" quick-start guide
- No architecture overview
- No link to deployment documentation
- No contribution guidelines link
- No license information

**Action Items**:
- [ ] Add project badges (license, build status)
- [ ] Add high-level architecture diagram
- [ ] Expand "What It Does" section with use cases
- [ ] Add "Quick Start" section for first-time users
- [ ] Add "Deployment" section with production guidance
- [ ] Link to CONTRIBUTING.md and LICENSE

### 6. Missing Dependencies Documentation
**Issue**: No clear documentation of:
- Python version requirements
- System dependencies (beyond what's in install script)
- Duckling version/commit used
- spaCy model version pinning

**Action Items**:
- [ ] Add `.python-version` file
- [ ] Pin Duckling to specific commit in documentation
- [ ] Document all system package requirements
- [ ] Consider adding `Pipfile.lock` or `poetry.lock`

---

## 🔧 Code Quality Issues

### 1. Missing Type Hints
**Severity**: LOW  
**Files**: Most Python files have incomplete type hints

**Recommendation**: Add comprehensive type hints for better IDE support and documentation.

### 2. No Linting Configuration
**Severity**: LOW  

**Missing Files**:
- `.pylintrc` or `pyproject.toml` with linting config
- `.editorconfig` for consistent formatting
- Pre-commit hooks configuration

**Action Items**:
- [ ] Add `.editorconfig`
- [ ] Add `pyproject.toml` with black/ruff/mypy configuration
- [ ] Consider adding pre-commit hooks

### 3. Test Coverage Unknown
**Issue**: No test coverage reporting configured.

**Action Items**:
- [ ] Add coverage.py configuration
- [ ] Add coverage badge to README
- [ ] Set coverage requirements in CI

---

## 🔒 Sensitive Data Audit Results

### ✅ No Hardcoded Secrets Found
- No API keys, passwords, or tokens in code
- No embedded credentials
- Environment variables properly used

### ✅ No Internal URLs Exposed
- No internal domain names found
- All URLs are configurable via environment

### ✅ Git History Clean
- No sensitive data in commit history
- No accidentally committed secrets

---

## 📦 Infrastructure-as-Code Gaps

### Missing IaC Templates
The repository is application-focused but requires significant AWS infrastructure:

**Required Resources** (not documented):
1. **ECS Cluster** with dedicated capacity
2. **EC2 Launch Templates** with custom UserData
3. **IAM Roles** (execution and task roles)
4. **ECR Repositories** (staging and production)
5. **CloudWatch Log Groups**
6. **VPC/Networking** configuration
7. **Load Balancers** (if applicable)
8. **Auto Scaling Groups** with placement constraints

**Recommendation**: Provide example IaC or link to infrastructure repository.

**Action Items**:
- [ ] Add `terraform/` or `cloudformation/` directory with examples
- [ ] Document infrastructure requirements in DEPLOYMENT.md
- [ ] Consider Docker Compose for local development

---

## 🎯 Recommended Pre-Release Checklist

### Phase 1: Security (Must Complete)
- [ ] Remove or template AWS account IDs
- [ ] Add .env.example with required variables
- [ ] Document GitHub secrets requirements
- [ ] Review and sanitize git history (if needed)
- [ ] Add SECURITY.md with vulnerability reporting

### Phase 2: Legal & Licensing (Must Complete)
- [ ] Choose and add LICENSE file
- [ ] Add copyright notices to major files
- [ ] Review third-party dependencies for license compatibility
- [ ] Add license badge to README

### Phase 3: Documentation (Strongly Recommended)
- [ ] Expand README with quick start guide
- [ ] Add CONTRIBUTING.md
- [ ] Add CODE_OF_CONDUCT.md
- [ ] Create architecture diagram
- [ ] Document infrastructure requirements
- [ ] Add deployment guide

### Phase 4: Code Quality (Nice to Have)
- [ ] Add linting configuration
- [ ] Set up test coverage reporting
- [ ] Add pre-commit hooks
- [ ] Add CI status badges
- [ ] Configure Dependabot for dependency updates

### Phase 5: Community (Optional but Recommended)
- [ ] Add issue templates (.github/ISSUE_TEMPLATE/)
- [ ] Add pull request template (.github/pull_request_template.md)
- [ ] Add GitHub Actions for automated testing
- [ ] Set up GitHub Discussions (optional)
- [ ] Add CHANGELOG.md

---

## 📋 Additional Recommendations

### 1. Repository Settings (Before Making Public)
- Enable branch protection for `main`
- Require pull request reviews
- Enable automated security alerts
- Configure Dependabot security updates
- Set up GitHub Actions permissions

### 2. Documentation Website (Future Enhancement)
Consider creating a documentation site using:
- GitHub Pages
- Read the Docs
- MkDocs

### 3. Release Process
- Set up semantic versioning
- Create release workflow
- Add CHANGELOG.md
- Tag releases properly

### 4. Docker Hub (Optional)
Consider publishing container images to Docker Hub for easier public access instead of requiring ECR setup.

---

## 🚀 Quick Start After Fixes

Once the above issues are addressed, the repository should include a clear path for external users:

1. **Clone & Install**: Simple setup instructions
2. **Local Development**: Docker Compose or local script
3. **Testing**: Run test suite easily
4. **Contribution**: Clear guidelines
5. **Deployment**: Infrastructure setup guide

---

## Summary

**High Priority** (Block Release):
- AWS account ID removal/templating
- LICENSE file addition
- Security documentation

**Medium Priority** (Should Fix):
- Comprehensive README update
- CONTRIBUTING.md
- Infrastructure documentation
- GitHub secrets documentation

**Low Priority** (Nice to Have):
- Code quality tooling
- Additional community files
- Advanced CI/CD features

**Estimated Effort**: 1-2 days of focused work to address critical and high-priority items.
