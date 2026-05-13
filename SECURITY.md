# Security Policy

This repository is a research IDS test suite for reproducible experimentation. Production deployments require operational hardening, model validation, logging review, and deployment controls.

## Reporting Issues

Please report security issues through GitHub Security Advisories or directly to the repository owner.

## Operational Notes

- Keep API keys out of source control. Configure AbuseIPDB with `ABUSEIPDB_API_KEY` or a local config file.
- Validate models against traffic from the intended environment before production use.
- Run Zeek and the monitor with the least privileges practical for your deployment.
