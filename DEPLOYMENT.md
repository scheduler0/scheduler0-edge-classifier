# Deployment Guide

This guide covers deploying the scheduler0-edge-classifier service to AWS ECS.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Task Definition Configuration](#task-definition-configuration)
- [GitHub Actions Setup](#github-actions-setup)
- [EC2 Bootstrap](#ec2-bootstrap)
- [Deployment](#deployment)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS ECS Cluster                      │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ EC2 Instance (Dedicated Capacity)                      │ │
│  │                                                          │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐   │ │
│  │  │ Host Process         │  │ ECS Task (Container) │   │ │
│  │  │                      │  │                       │   │ │
│  │  │ Duckling (Port 8000) │◄─┤ FastAPI App          │   │ │
│  │  │ (Time Parser)        │  │ (Port 8080)          │   │ │
│  │  └──────────────────────┘  │                       │   │ │
│  │                             │ - spaCy NLP          │   │ │
│  │  /opt/duckling/            │ - Intent Classifier  │   │ │
│  │  /opt/scheduler0-nlp/      │ - Suggestions API    │   │ │
│  │                             └──────────────────────┘   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Application Load Balancer
                              │
                              ▼
                          Internet
```

### Key Design Decisions

1. **Duckling runs on the EC2 host**, not in containers
   - Haskell GHC RTS has issues with Docker time parsing
   - Built once during EC2 bootstrap, shared across containers

2. **Python virtualenv is host-mounted**
   - Heavy NLP dependencies installed during bootstrap
   - Shared read-only across containers
   - Keeps container images small

3. **Dedicated EC2 capacity**
   - Uses ECS placement constraints
   - Ensures NLP models are pre-loaded
   - Faster cold starts

## Prerequisites

### AWS Resources Required

1. **IAM Roles**
   - ECS Task Execution Role
   - ECS Task Role
   - Permissions: ECR, CloudWatch Logs, ECS

2. **VPC & Networking**
   - VPC with private subnets
   - Security groups for ECS tasks
   - NAT Gateway for external access

3. **ECS Cluster**
   - EC2-backed cluster (not Fargate)
   - Dedicated EC2 instances with custom attributes

4. **ECR Repositories**
   - `staging-scheduler0-intent-classifier`
   - `production-scheduler0-intent-classifier`

5. **CloudWatch Log Groups**
   - `ecs/staging-scheduler0-intent-classifier-ecs-service`
   - `ecs/production-scheduler0-intent-classifier-ecs-service`

### Local Requirements

- AWS CLI configured
- Docker installed
- GitHub repository access
- Python 3.9+

## Infrastructure Setup

### Option 1: Manual AWS Setup

#### 1. Create IAM Roles

**Task Execution Role** (`staging_scheduler0_ecs_task_execution_role`):
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
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

**Trust Policy**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

#### 2. Create ECR Repositories

```bash
aws ecr create-repository \
  --repository-name staging-scheduler0-intent-classifier \
  --region us-east-1

aws ecr create-repository \
  --repository-name production-scheduler0-intent-classifier \
  --region us-east-1
```

#### 3. Create CloudWatch Log Groups

```bash
aws logs create-log-group \
  --log-group-name ecs/staging-scheduler0-intent-classifier-ecs-service \
  --region us-east-1

aws logs create-log-group \
  --log-group-name ecs/production-scheduler0-intent-classifier-ecs-service \
  --region us-east-1
```

#### 4. Create ECS Cluster

```bash
aws ecs create-cluster \
  --cluster-name staging_scheduler0_ecs_cluster \
  --region us-east-1
```

#### 5. Launch EC2 Instances

Create a launch template with **custom UserData** to set the cluster attribute:

```bash
#!/bin/bash
echo ECS_CLUSTER=staging_scheduler0_ecs_cluster >> /etc/ecs/ecs.config
echo ECS_INSTANCE_ATTRIBUTES='{"scheduler0.intent-classifier":"true"}' >> /etc/ecs/ecs.config
```

**Important**: Use Amazon Linux 2023 AMI for compatibility.

### Option 2: Terraform (Example)

```hcl
# terraform/main.tf
resource "aws_ecs_cluster" "scheduler0" {
  name = "${var.environment}_scheduler0_ecs_cluster"
}

resource "aws_ecr_repository" "intent_classifier" {
  name = "${var.environment}-scheduler0-intent-classifier"
}

resource "aws_cloudwatch_log_group" "intent_classifier" {
  name              = "ecs/${var.environment}-scheduler0-intent-classifier-ecs-service"
  retention_in_days = 7
}

resource "aws_launch_template" "ecs_instance" {
  name_prefix   = "${var.environment}-scheduler0-ecs-"
  image_id      = data.aws_ami.ecs_optimized.id
  instance_type = "t3.medium"

  iam_instance_profile {
    name = aws_iam_instance_profile.ecs_instance.name
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name = aws_ecs_cluster.scheduler0.name
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.environment}-scheduler0-intent-classifier"
    }
  }
}

# ... additional resources (security groups, IAM, etc.)
```

## Task Definition Configuration

### Generate Task Definition from Template

1. **Copy environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your AWS account ID and region
   ```

2. **Generate task definition** (using envsubst or script):
   ```bash
   export AWS_ACCOUNT_ID="123456789012"
   export AWS_REGION="us-east-1"
   
   envsubst < staging-scheduler0-intent-classifier-ecs-task-definition.template.json \
     > staging-scheduler0-intent-classifier-ecs-task-definition.json
   ```

3. **Validate task definition**:
   ```bash
   aws ecs register-task-definition \
     --cli-input-json file://staging-scheduler0-intent-classifier-ecs-task-definition.json \
     --region us-east-1
   ```

### Task Definition Key Settings

- **CPU**: 1024 (1 vCPU)
- **Memory**: 2048 MB
- **Network Mode**: `awsvpc`
- **Placement Constraint**: `attribute:scheduler0.intent-classifier == true`
- **Health Check**: `/healthz` endpoint with 300s start period

## GitHub Actions Setup

### Required Secrets

Configure in GitHub Settings → Secrets and variables → Actions:

| Secret Name | Description |
|------------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key with ECR/ECS permissions |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |

### Required Variables

| Variable Name | Value |
|--------------|-------|
| `AWS_REGION` | `us-east-1` (or your region) |

### IAM Permissions for GitHub Actions

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
        "ecs:DescribeTaskDefinition"
      ],
      "Resource": "*"
    }
  ]
}
```

## EC2 Bootstrap

The EC2 instances require a one-time bootstrap to install:
- Duckling (Haskell time parser)
- Python NLP virtualenv
- spaCy models

### Run Bootstrap Script

SSH into your EC2 instance and run:

```bash
curl -O https://raw.githubusercontent.com/scheduler0/scheduler0-edge-classifier/main/install_scheduler0_nlp.sh
chmod +x install_scheduler0_nlp.sh
./install_scheduler0_nlp.sh --ci
```

**⏱️ Bootstrap Time**: 30-90 minutes on first run (compiles Duckling from source)

### Automated Bootstrap (UserData)

Add to EC2 launch template UserData (after ECS config):

```bash
#!/bin/bash
# ... ECS config above ...

# Bootstrap NLP dependencies
curl -O https://raw.githubusercontent.com/scheduler0/scheduler0-edge-classifier/main/install_scheduler0_nlp.sh
chmod +x install_scheduler0_nlp.sh
./install_scheduler0_nlp.sh --ci 2>&1 | tee /var/log/scheduler0-intent-classifier-bootstrap.log
```

### Verify Bootstrap

```bash
# Check bootstrap completion
cat /var/lib/scheduler0-intent-classifier/bootstrap-complete

# Verify Duckling binary
/opt/duckling/duckling-example-exe --help

# Verify Python venv
/opt/scheduler0-nlp/bin/python --version
/opt/scheduler0-nlp/bin/python -c "import spacy; print(spacy.load('en_core_web_sm'))"
```

### Start Duckling Service

Create a systemd service to keep Duckling running:

```ini
# /etc/systemd/system/duckling.service
[Unit]
Description=Duckling Time Parser
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/duckling
ExecStart=/opt/duckling/duckling-example-exe
Restart=always
RestartSec=10
Environment="LD_LIBRARY_PATH=/opt/duckling/lib"

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
systemctl daemon-reload
systemctl enable duckling
systemctl start duckling

# Write host IP for containers
echo $(hostname -I | awk '{print $1}') > /opt/duckling/host_ip
```

## Deployment

### Automatic Deployment (GitHub Actions)

Push to trigger deployment:

```bash
# Deploy to staging
git push origin staging

# Deploy to production
git push origin main
```

The workflow will:
1. Build Docker image
2. Push to ECR
3. Register new task definition
4. Update ECS service
5. Wait for deployment (up to 3 minutes)

### Manual Deployment

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t scheduler0-intent-classifier .

# Tag and push
docker tag scheduler0-intent-classifier:latest \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/staging-scheduler0-intent-classifier:latest

docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/staging-scheduler0-intent-classifier:latest

# Update service
aws ecs update-service \
  --cluster staging_scheduler0_ecs_cluster \
  --service staging_scheduler0-intent-classifier-service \
  --force-new-deployment \
  --region us-east-1
```

## Verification

### Health Check

```bash
# Get ECS task public IP (if in public subnet) or use load balancer
TASK_IP="<TASK_IP_OR_ALB>"

curl http://${TASK_IP}:8080/healthz
```

Expected response:
```json
{"status": "ok"}
```

### Classify Intent

```bash
curl -X POST http://${TASK_IP}:8080/v1/intents/classify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Remind me every Monday at 9am"}'
```

### Check ECS Service Status

```bash
aws ecs describe-services \
  --cluster staging_scheduler0_ecs_cluster \
  --services staging_scheduler0-intent-classifier-service \
  --region us-east-1 \
  --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'
```

### View Logs

```bash
# Get log stream name
aws logs describe-log-streams \
  --log-group-name ecs/staging-scheduler0-intent-classifier-ecs-service \
  --order-by LastEventTime \
  --descending \
  --max-items 1 \
  --region us-east-1

# Tail logs
aws logs tail ecs/staging-scheduler0-intent-classifier-ecs-service \
  --follow \
  --region us-east-1
```

## Troubleshooting

### Container Won't Start

**Symptom**: Tasks immediately stop or fail health checks

**Check**:
1. Verify bootstrap completed:
   ```bash
   ssh ec2-user@<EC2_IP>
   cat /var/lib/scheduler0-intent-classifier/bootstrap-complete
   tail -100 /var/log/scheduler0-intent-classifier-bootstrap.log
   ```

2. Verify Duckling is running:
   ```bash
   systemctl status duckling
   curl http://127.0.0.1:8000/parse -d 'locale=en_GB&text=tomorrow&dims=["time"]'
   ```

3. Check ECS task logs:
   ```bash
   aws logs tail ecs/staging-scheduler0-intent-classifier-ecs-service --follow
   ```

### Duckling Connection Failed

**Symptom**: Health check fails with "Duckling unreachable"

**Solutions**:
1. Ensure Duckling service is running on host
2. Check `/opt/duckling/host_ip` file exists and is correct
3. Verify container can reach host IP on port 8000
4. Check security group allows traffic from container to host

### Bootstrap Takes Too Long

**Expected**: 30-90 minutes on first run

**To speed up**:
- Use pre-built AMI with bootstrap complete
- Create golden AMI after successful bootstrap
- Use larger instance type during bootstrap (then scale down)

### Health Check Fails Immediately

**Symptom**: Task fails health check at startup

**Note**: Task definition uses `startPeriod: 300` (5 minutes) to allow for:
- Container download
- spaCy model loading
- Duckling connection establishment

If failing after 5 minutes, check application logs.

### Placement Constraint Not Satisfied

**Symptom**: Tasks remain in PENDING state

**Solution**: Ensure EC2 instance has the correct attribute:
```bash
# On EC2 instance
cat /etc/ecs/ecs.config | grep INSTANCE_ATTRIBUTES
# Should show: ECS_INSTANCE_ATTRIBUTES={"scheduler0.intent-classifier":"true"}

# Verify in AWS
aws ecs describe-container-instances \
  --cluster staging_scheduler0_ecs_cluster \
  --container-instances <CONTAINER_INSTANCE_ARN> \
  --query 'containerInstances[0].attributes'
```

## Production Considerations

### Scaling

Current setup uses dedicated EC2 instances. For scaling:

1. **Horizontal**: Add more EC2 instances to the cluster
2. **Vertical**: Use larger instance types (c5.xlarge, etc.)
3. **Task Count**: Increase desired task count in service

### Monitoring

Set up CloudWatch alarms for:
- CPU/Memory utilization
- Health check failures
- 5xx error rate
- Response time (p50, p95, p99)

### High Availability

- Deploy across multiple availability zones
- Use Application Load Balancer
- Set up CloudWatch alarms
- Enable auto-scaling based on CPU/memory

### Security

- Keep EC2 instances in private subnets
- Use security groups to restrict access
- Rotate IAM credentials regularly
- Enable VPC flow logs
- Use AWS WAF if exposing publicly

### Backup & Disaster Recovery

- Tag EC2 instances for automated snapshots
- Document infrastructure as code
- Keep task definitions in version control
- Test recovery procedures regularly

## Additional Resources

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [Duckling GitHub](https://github.com/facebook/duckling)
- [spaCy Models](https://spacy.io/models)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
