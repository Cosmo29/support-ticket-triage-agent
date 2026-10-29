# GDPR, Privacy Compliance, and Data Export Protocols

## Overview
Guidelines for handling customer data privacy requests under GDPR (Articles 15 & 17), CCPA, and enterprise compliance audits.

## Supported Request Types
1. **Right to Access / Data Portability (Article 20)**:
   - Full JSON / CSV export of workspace tickets, user profiles, audit logs, and uploaded attachments.
   - Self-service trigger: `Admin Console > Organization Settings > Compliance > Request Data Archive`.
   - Generation turnaround: automated within 4 to 24 hours depending on archive size.
2. **Right to Erasure / "To Be Forgotten" (Article 17)**:
   - Complete redaction and permanent cryptographic deletion of personal identifiers (PII).
   - System retention override: Billing transaction records must be retained for 7 years per statutory tax requirements, but all customer data and ticket transcripts are purged within 30 days.

## Agent Action Checklist
- Authenticate requester as verified primary Organization Owner.
- Cross-reference with Legal & Compliance queue before executing destructive deletion operations.
- Generate and transmit tamper-proof certificate of erasure upon job completion.
