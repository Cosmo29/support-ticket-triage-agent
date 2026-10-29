# Authentication, Single Sign-On (SSO), and 2FA Troubleshooting

## Overview
Procedures for resolving SAML 2.0, Okta, Azure AD / Entra ID, Google Workspace SSO integration errors, and multi-factor authentication (2FA) lockouts.

## Common SSO Issues
1. **SAML Certificate Expiration (`SAML_CERT_EXPIRED`)**:
   - Symptoms: All workspace users receive "Invalid SAML Signature" or HTTP 403 on login redirect.
   - Fix: Workspace admin must upload renewed X.509 certificate under `Admin Console > Security > SSO > SAML Certificate`.
2. **Entity ID or Assertion Consumer Service (ACS) URL Mismatch**:
   - Verification: ACS URL must exactly match `https://auth.platform.io/saml/consume/<workspace_id>`.
   - Entity ID is case-sensitive: `https://auth.platform.io/saml/metadata`.
3. **SCIM Provisioning Sync Errors**:
   - Error 409 Conflict: User already exists with a different external identity ID.
   - Error 400 Bad Request: Missing required claims (`email`, `firstName`, `lastName`).

## 2FA / MFA Account Recovery Protocol
- **Self-Service Recovery**: Users can use one of their 10 emergency 16-character backup recovery codes generated at setup.
- **Admin Reset**: Workspace owners can reset a member's 2FA device from `Admin > Members > [Select Member] > Reset MFA`.
- **Identity Verification**: If workspace owner is locked out, require 2-point identity verification (government ID + verified billing card on file) before manual recovery token generation. Never bypass without manager approval.
