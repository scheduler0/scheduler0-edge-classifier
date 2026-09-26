# GitHub Actions CI/CD Documentation

This document explains the GitHub Actions workflows configured for this repository.

## Table of Contents

- [Overview](#overview)
- [Workflows](#workflows)
- [Required Secrets](#required-secrets)
- [Setup Instructions](#setup-instructions)
- [Workflow Details](#workflow-details)
- [Troubleshooting](#troubleshooting)

## Overview

The repository uses GitHub Actions for automated deployment to AWS ECS. There are two deployment environments:

- **Staging**: Automatically deploys on push to `staging` branch
- **Production**: Automatically deploys on push to `main` branch

## Workflows

### 1. Staging Deployment

**File**: `.github/workflows/deploy-staging.yml`  
**Trigger**: Push to `staging` branch or manual dispatch  
**Target**: Staging ECS cluster  

### 2. Production Deployment

**File**: `.github/workflows/deploy.yml`  
**Trigger**: Push to `main` branch or manual dispatch  
**Target**: Production ECS cluster  

### 3. Reusable Deployment Workflow

**File**: `.github/workflows/_deploy.yml`  
**Purpose**: Shared deployment logic used by staging and production workflows  
**Runs**: Called by other workflows, not directly triggered  

## Required Secrets

Configure these in **GitHub Settings → Secrets and variables → Actions**:

### Secrets

| Secret Name | Description | Example | Required |
|------------|-------------|---------|----------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key | `AKIAIOSFODNN7EXAMPLE` | ✅ |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` | ✅ |

### Variables

| Variable Name | Description | Example | Required |
|--------------|-------------|---------|----------|
| `AWS_REGION` | AWS region for deployment | `us-east-1` | ✅ |

## Setup Instructions

### Step 1: Create IAM User for GitHub Actions

1. **Go to AWS IAM Console**
2. **Create a new IAM user** (e.g., `github-actions-scheduler0-classifier`)
3. **Attach the following policy**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:RegisterTaskDefinition",
        "ecs:DescribeTaskDefinition",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

4. **Create access keys** for this user
5. **Save the access key ID and secret** (you'll need them next)

### Step 2: Add Secrets to GitHub

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret:
   - **Name**: `AWS_ACCESS_KEY_ID`  
     **Value**: (paste your access key ID)
   - **Name**: `AWS_SECRET_ACCESS_KEY`  
     **Value**: (paste your secret access key)

### Step 3: Add Variables to GitHub

1. In the same section, go to the **Variables** tab
2. Click **New repository variable**
3. Add:
   - **Name**: `AWS_REGION`  
     **Value**: `us-east-1` (or your region)

### Step 4: Verify ECS Infrastructure

Ensure the following resources exist in your AWS account:

**For Staging**:
- ECS Cluster: `staging_scheduler0_ecs_cluster`
- ECS Service: `staging_scheduler0-intent-classifier-service`
- ECR Repository: `staging-scheduler0-intent-classifier`
- Task Definition: `staging_scheduler0-intent-classifier`
- CloudWatch Log Group: `ecs/staging-scheduler0-intent-classifier-ecs-service`

**For Production**:
- ECS Cluster: `production_scheduler0_ecs_cluster`
- ECS Service: `production_scheduler0-intent-classifier-service`
- ECR Repository: `production-scheduler0-intent-classifier`
- Task Definition: `production_scheduler0-intent-classifier`
- CloudWatch Log Group: `ecs/production-scheduler0-intent-classifier-ecs-service`

See [DEPLOYMENT.md](DEPLOYMENT.md) for infrastructure setup instructions.

## Workflow Details

### Deployment Process

The deployment workflow performs these steps:

1. **Checkout code** from the repository
2. **Derive resource names** based on environment (staging/production)
3. **Configure AWS credentials** using GitHub secrets
4. **Login to Amazon ECR** for Docker image push
5. **Build Docker image** using the Dockerfile
6. **Tag and push image** to ECR
   - Tags: `<git-sha>` and `latest`
7. **Render task definition** with new image URI
8. **Register new task definition** with ECS
9. **Update ECS service** to use new task definition
10. **Wait for service stabilization** (up to 3 minutes)
11. **Output deployment status**

### Workflow Outputs

The workflow logs:
- Image URI that was pushed to ECR
- Task definition ARN that was registered
- Service deployment status
- Running/desired task counts

### Concurrency

Each environment has concurrency control:
```yaml
concurrency:
  group: intent-classifier-deploy-staging-${{ github.ref }}
  cancel-in-progress: true
```

This ensures only one deployment runs at a time per branch, canceling in-progress runs when new commits are pushed.

### Service Stabilization

The workflow waits up to 3 minutes for the ECS service to stabilize. This timeout is intentional because:

- EC2 instances may still be bootstrapping (NLP install takes 60-90 minutes on first run)
- Container download and startup can take time
- Health checks have a 5-minute start period

If the service doesn't stabilize, the workflow logs a warning but **does not fail**. CloudWatch alarms are the authoritative signal for service health.

## Triggering Deployments

### Automatic (on push)

```bash
# Deploy to staging
git push origin staging

# Deploy to production
git push origin main
```

### Manual (workflow dispatch)

1. Go to **Actions** tab in GitHub
2. Select the workflow (staging or production)
3. Click **Run workflow**
4. Select the branch
5. Click **Run workflow**

### Environment Protection Rules

Consider setting up environment protection rules in GitHub:

1. Go to **Settings** → **Environments**
2. Create `staging` and `production` environments
3. Add protection rules:
   - **Required reviewers**: Require approval before production deployments
   - **Wait timer**: Add a delay before deployment
   - **Branch protection**: Limit to specific branches

## Monitoring Deployments

### Via GitHub Actions UI

1. Go to **Actions** tab
2. Click on the workflow run
3. Expand steps to see detailed logs

### Via AWS Console

```bash
# Check ECS service status
aws ecs describe-services \
  --cluster staging_scheduler0_ecs_cluster \
  --services staging_scheduler0-intent-classifier-service \
  --region us-east-1

# View recent deployments
aws ecs describe-services \
  --cluster staging_scheduler0_ecs_cluster \
  --services staging_scheduler0-intent-classifier-service \
  --region us-east-1 \
  --query 'services[0].deployments'

# Check task status
aws ecs list-tasks \
  --cluster staging_scheduler0_ecs_cluster \
  --service-name staging_scheduler0-intent-classifier-service \
  --region us-east-1
```

### Via CloudWatch Logs

View application logs:
```bash
aws logs tail ecs/staging-scheduler0-intent-classifier-ecs-service \
  --follow \
  --region us-east-1
```

## Troubleshooting

### Deployment Fails at "Login to Amazon ECR"

**Symptom**: Error message about authentication

**Solution**:
- Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct
- Ensure IAM user has `ecr:GetAuthorizationToken` permission
- Check if access keys are still active in IAM

### Deployment Fails at "Register task definition"

**Symptom**: Error registering task definition

**Solution**:
- Verify IAM user has `ecs:RegisterTaskDefinition` and `iam:PassRole` permissions
- Check that execution role ARNs in task definition exist
- Ensure task definition JSON is valid

### Service Does Not Stabilize

**Symptom**: Warning about service not stabilizing within 3 minutes

**This is expected** in these scenarios:
- First deployment (EC2 bootstrap in progress)
- Cold start after instance replacement
- Container image is large

**Check**:
```bash
# View ECS service events
aws ecs describe-services \
  --cluster staging_scheduler0_ecs_cluster \
  --services staging_scheduler0-intent-classifier-service \
  --query 'services[0].events[0:10]'

# Check if tasks are running
aws ecs list-tasks \
  --cluster staging_scheduler0_ecs_cluster \
  --service-name staging_scheduler0-intent-classifier-service
```

### Tasks Stuck in PENDING State

**Symptom**: Desired count shows tasks, but running count is 0

**Possible Causes**:
1. **No EC2 instances available** with the required attribute
2. **Bootstrap not complete** on EC2 instance
3. **Placement constraint not satisfied**

**Solution**:
```bash
# Check container instances
aws ecs list-container-instances \
  --cluster staging_scheduler0_ecs_cluster

# Check instance attributes
aws ecs describe-container-instances \
  --cluster staging_scheduler0_ecs_cluster \
  --container-instances <ARN> \
  --query 'containerInstances[0].attributes'

# Should show: {"name": "scheduler0.intent-classifier", "value": "true"}
```

### Image Push Fails

**Symptom**: Error pushing image to ECR

**Solution**:
- Verify ECR repository exists
- Check IAM permissions for ECR push actions
- Ensure repository policies allow the IAM user

### Workflow Not Triggering

**Symptom**: Push to `staging` or `main` doesn't trigger workflow

**Solution**:
- Check workflow YAML syntax (Actions tab shows errors)
- Verify branch names match exactly (case-sensitive)
- Ensure workflows are enabled (Settings → Actions → General)

## Best Practices

### 1. Test in Staging First

Always deploy to staging before production:
```bash
git checkout staging
git merge feature-branch
git push origin staging
# Verify in staging, then merge to main
```

### 2. Use Pull Requests

- Require PR reviews before merging to `main`
- Run tests in CI before allowing merge
- Use branch protection rules

### 3. Monitor Deployments

- Watch the GitHub Actions logs during deployment
- Check CloudWatch alarms after deployment
- Verify health check endpoint responds

### 4. Rollback Strategy

If a deployment fails:

**Option A: Revert code**
```bash
git revert <commit-sha>
git push origin main  # Triggers new deployment
```

**Option B: Manual ECS rollback**
```bash
# Find previous task definition
aws ecs list-task-definitions \
  --family staging_scheduler0-intent-classifier \
  --sort DESC

# Update service to previous revision
aws ecs update-service \
  --cluster staging_scheduler0_ecs_cluster \
  --service staging_scheduler0-intent-classifier-service \
  --task-definition staging_scheduler0-intent-classifier:123
```

### 5. Secrets Rotation

Rotate AWS credentials regularly:
1. Create new access key in IAM
2. Update GitHub secrets
3. Delete old access key
4. Test deployment with new keys

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [DEPLOYMENT.md](DEPLOYMENT.md) - Full deployment guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
