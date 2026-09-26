# Open Source Release Checklist

Use this checklist before making the repository public.

## ✅ Completed (via PR #7)

### Security
- [x] Remove AWS account IDs from repository
- [x] Create task definition templates with placeholders
- [x] Add `.env.example` for configuration guidance
- [x] Update `.gitignore` to exclude sensitive files
- [x] Add `SECURITY.md` with vulnerability reporting

### Legal & Licensing
- [x] Add `LICENSE` file (MIT License)
- [x] Review third-party dependency licenses
- [x] Add license badge to README

### Documentation
- [x] Add comprehensive `README.md`
- [x] Create `CONTRIBUTING.md`
- [x] Create `DEPLOYMENT.md`
- [x] Add `GITHUB_ACTIONS.md` for CI/CD
- [x] Create issue and PR templates
- [x] Add `.editorconfig` for consistency

### Code Quality
- [x] Review git history for sensitive data
- [x] Ensure no hardcoded secrets
- [x] Verify no internal URLs or references

## ⏳ Pre-Release Actions (Do Before Making Public)

### GitHub Repository Settings

- [ ] **Go to Settings → General**
  - [ ] Update repository description
  - [ ] Add website URL (if applicable)
  - [ ] Add topics/tags (e.g., `nlp`, `intent-classification`, `spacy`, `fastapi`, `scheduler`)
  - [ ] Enable features: Issues, Discussions (optional), Projects (optional)
  - [ ] Disable: Wiki (unless you want to use it)

- [ ] **Settings → Branches**
  - [ ] Add branch protection rule for `main`:
    - [ ] Require pull request reviews (at least 1)
    - [ ] Require status checks to pass
    - [ ] Require conversation resolution before merging
    - [ ] Do not allow bypassing the above settings

- [ ] **Settings → Security**
  - [ ] Enable Dependabot alerts
  - [ ] Enable Dependabot security updates
  - [ ] Enable secret scanning (if available)
  - [ ] Enable code scanning (GitHub Advanced Security)

- [ ] **Settings → Actions → General**
  - [ ] Set "Allow all actions and reusable workflows" or restrict as needed
  - [ ] Enable "Allow GitHub Actions to create and approve pull requests" if using automation

### GitHub Secrets/Variables

- [ ] **Verify all required secrets are set**:
  - [ ] `AWS_ACCESS_KEY_ID`
  - [ ] `AWS_SECRET_ACCESS_KEY`

- [ ] **Verify all required variables are set**:
  - [ ] `AWS_REGION`

- [ ] **Review secret access**:
  - [ ] Ensure secrets are only accessible to necessary workflows
  - [ ] Consider using environment-specific secrets

### Code Review

- [ ] **Final security scan**:
  ```bash
  # Search for any remaining sensitive patterns
  git grep -i "aws_account_id" -- ':!*.template.json' ':!*.md'
  git grep -i "748201447723"
  git grep -iE "(password|secret|key|token)" -- ':!*.md' ':!*.example'
  ```

- [ ] **Check for internal references**:
  ```bash
  # Look for internal URLs or service names
  git grep -i "internal\."
  git grep -i "\.local"
  ```

- [ ] **Verify .gitignore is working**:
  ```bash
  git status --ignored
  # Should show real task definition files as ignored
  ```

### Infrastructure Validation

- [ ] **Confirm AWS resources are ready**:
  - [ ] ECS clusters exist (staging, production)
  - [ ] ECR repositories exist
  - [ ] IAM roles configured
  - [ ] CloudWatch log groups created
  - [ ] EC2 instances bootstrapped

- [ ] **Test deployment workflow**:
  - [ ] Staging deployment works
  - [ ] Production deployment works (or is ready)

### Documentation Final Check

- [ ] **README.md**:
  - [ ] All badges work
  - [ ] All links are valid
  - [ ] Quick start instructions tested
  - [ ] Examples are accurate

- [ ] **CONTRIBUTING.md**:
  - [ ] Setup instructions tested
  - [ ] Contribution workflow documented
  - [ ] Code style guidelines clear

- [ ] **DEPLOYMENT.md**:
  - [ ] Architecture diagram accurate
  - [ ] All required resources documented
  - [ ] Instructions tested

- [ ] **SECURITY.md**:
  - [ ] Contact email is correct (or use GitHub security advisories)
  - [ ] Supported versions listed
  - [ ] Response timeline is reasonable

### Testing

- [ ] **Run full test suite**:
  ```bash
  python -m unittest discover -s tests -v
  ```

- [ ] **Test local development setup**:
  ```bash
  ./run-local.sh
  # Verify API responds correctly
  ```

- [ ] **Verify Docker build**:
  ```bash
  docker build -t test-classifier .
  ```

## 🚀 Making Repository Public

### Step 1: Final Preparation

- [ ] Merge PR #7 (open source preparation)
- [ ] Create a release tag:
  ```bash
  git tag -a v1.0.0 -m "Initial open source release"
  git push origin v1.0.0
  ```
- [ ] Create GitHub release from tag with release notes

### Step 2: Make Public

- [ ] **Go to Settings → Danger Zone**
- [ ] **Click "Change repository visibility"**
- [ ] **Select "Public"**
- [ ] **Type repository name to confirm**
- [ ] **Click "I understand, change repository visibility"**

### Step 3: Post-Release Actions

- [ ] **Announce the release**:
  - [ ] Tweet/social media (if applicable)
  - [ ] Blog post (if applicable)
  - [ ] Update related documentation
  - [ ] Notify stakeholders

- [ ] **Monitor initial response**:
  - [ ] Watch for issues
  - [ ] Respond to questions promptly
  - [ ] Be prepared for feedback

- [ ] **Set up notifications**:
  - [ ] Watch repository for issues/PRs
  - [ ] Configure email notifications
  - [ ] Set up Slack/Discord integration (if applicable)

## 📊 Post-Launch Checklist

### Week 1

- [ ] Respond to all issues/PRs within 48 hours
- [ ] Monitor security alerts
- [ ] Check GitHub Insights for traffic
- [ ] Review any automated Dependabot PRs

### Ongoing

- [ ] **Monthly**:
  - [ ] Review and triage open issues
  - [ ] Update dependencies
  - [ ] Check security advisories
  - [ ] Review analytics

- [ ] **Quarterly**:
  - [ ] Rotate AWS credentials
  - [ ] Review contributor activity
  - [ ] Update documentation if needed
  - [ ] Create releases for significant changes

- [ ] **Annually**:
  - [ ] Review project direction
  - [ ] Update copyright years
  - [ ] Evaluate license choice
  - [ ] Consider roadmap for next year

## 🆘 Rollback Plan

If you need to make the repository private again:

1. **Go to Settings → Danger Zone**
2. **Click "Change repository visibility"**
3. **Select "Private"**
4. **Confirm the change**

**Note**: This won't delete any forks that were made while the repo was public.

## 📚 Additional Resources

- [GitHub: Managing repository visibility](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)
- [GitHub: Securing your repository](https://docs.github.com/en/code-security/getting-started/securing-your-repository)
- [Open Source Guide](https://opensource.guide/)
- [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/)

---

**Questions or issues?** Review `OPEN_SOURCE_AUDIT.md` for detailed analysis and recommendations.
