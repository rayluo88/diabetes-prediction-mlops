# Security Policy

## Known Vulnerabilities

### MLflow Deserialization Vulnerabilities (CVE-2024-37052 to CVE-2024-37060)

**Status**: UNPATCHED - No fix available as of August 2025

**Affected Component**: MLflow (all versions)

**Vulnerability Type**: Unsafe deserialization leading to Remote Code Execution

**CVSS Score**: 8.8 (High)

**Description**: Multiple vulnerabilities in MLflow allow deserialization of untrusted data through malicious model files. An attacker can inject malicious pickle objects into model files that execute arbitrary code when loaded.

**Affected CVEs**:
- CVE-2024-37059 (PyTorch models)
- CVE-2024-37057 (TensorFlow models) 
- CVE-2024-37052, CVE-2024-37053, CVE-2024-37054, CVE-2024-37055, CVE-2024-37056, CVE-2024-37060 (Various model formats)

### Mitigation Strategy

Since no patched version is available, we implement the following mitigations:

1. **Trust Boundary**: Only load models from trusted sources
2. **Access Control**: Implement strict access controls for model upload/download
3. **Monitoring**: Enhanced logging and monitoring of model operations
4. **Isolation**: Run MLflow in isolated environments with limited privileges
5. **Regular Updates**: Monitor for security patches and update immediately when available

### Production Deployment Recommendations

For production deployments of this diabetes prediction system:

1. **Network Isolation**: Deploy MLflow behind firewalls with restricted access
2. **User Authentication**: Implement strong authentication for all MLflow access
3. **Model Validation**: Implement additional validation for uploaded models
4. **Audit Logging**: Maintain comprehensive audit logs for all model operations
5. **Regular Security Reviews**: Conduct regular security assessments

### Reporting Security Issues

If you discover a security vulnerability in this project, please:

1. Do NOT create a public GitHub issue
2. Email the security team with details
3. Allow time for assessment and patching before public disclosure

### Security Updates

This document will be updated as:
- New vulnerabilities are discovered
- Patches become available
- Mitigation strategies are enhanced

Last updated: August 2025