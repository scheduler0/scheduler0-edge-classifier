# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of scheduler0-edge-classifier seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **Preferred**: Use GitHub's private vulnerability reporting feature
   - Go to the Security tab of this repository
   - Click "Report a vulnerability"
   - Fill out the form with details

2. **Alternative**: Send an email to security@scheduler0.com (if you have a security contact email)

### What to Include

Please include the following information in your report:

- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- We will acknowledge receipt of your vulnerability report within 3 business days
- We will send a more detailed response within 7 business days indicating the next steps
- We will keep you informed about the progress toward a fix and announcement
- We may ask for additional information or guidance

## Security Best Practices for Users

When deploying this service:

1. **Secrets Management**
   - Never commit AWS credentials to version control
   - Use IAM roles for ECS tasks instead of access keys when possible
   - Rotate credentials regularly
   - Use AWS Secrets Manager or SSM Parameter Store for sensitive configuration

2. **Network Security**
   - Deploy the service in a private subnet
   - Use security groups to restrict access to necessary ports only
   - Consider using a VPN or bastion host for administrative access
   - Use AWS WAF if exposing the service publicly

3. **Infrastructure**
   - Keep EC2 instances and container images updated
   - Enable CloudWatch logging and monitoring
   - Use least-privilege IAM policies
   - Enable AWS GuardDuty for threat detection

4. **Application**
   - Keep Python dependencies updated
   - Monitor for security advisories for spaCy, FastAPI, and other dependencies
   - Validate and sanitize all user input
   - Implement rate limiting to prevent abuse

## Known Security Considerations

### Dependencies

This project relies on several external dependencies:

- **spaCy**: NLP library - monitor for security updates
- **FastAPI/Uvicorn**: Web framework - keep updated
- **Duckling**: Time parsing - uses Haskell runtime
- **Python 3.x**: Requires security updates from your OS

### Data Privacy

- The service processes user text for intent classification
- Consider data retention policies for logs
- Be aware of PII in user messages
- Implement appropriate data protection measures for your use case

## Security Update Process

When a security vulnerability is fixed:

1. A security advisory will be published on GitHub
2. A patch release will be created and tagged
3. Release notes will include security fix information
4. Users will be notified through GitHub watch notifications

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged in release notes (unless they prefer to remain anonymous).
