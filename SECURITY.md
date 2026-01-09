# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Semantic Unit, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email the maintainer directly at: **chaitanyakumar435@outlook.com**
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- **Acknowledgment**: Within 48 hours of your report
- **Initial Assessment**: Within 1 week
- **Resolution Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

### Scope

Security issues we're interested in:

- API key exposure or leakage
- Injection vulnerabilities in LLM prompts
- Dependencies with known vulnerabilities
- Authentication/authorization bypasses
- Data exposure issues

### Out of Scope

- Issues in third-party dependencies (report to those projects directly)
- Denial of service through normal usage
- Social engineering attacks
- Physical security issues

### Recognition

We appreciate security researchers who help keep Semantic Unit secure. With your permission, we'll acknowledge your contribution in our release notes.

## Security Best Practices

When using Semantic Unit:

1. **Never commit API keys** - Use environment variables or `.env` files
2. **Keep dependencies updated** - Run `pip install --upgrade semantic-unit`
3. **Use minimal permissions** - Only grant necessary API access
4. **Monitor usage** - Track API calls and costs

## Contact

For security concerns: chaitanyakumar435@outlook.com

For general issues: [GitHub Issues](https://github.com/chaitanyakumar-d/semantic-unit/issues)
