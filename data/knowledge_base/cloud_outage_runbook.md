# Cloud Outage, Service Degradation, and Incident Response

## Severity Classification
- **P0 - Blocker / Total Outage**: Core platform unreachable, data pipeline completely stalled, or catastrophic data loss risk affecting multiple customers. Immediate incident commander page. SLA response: < 15 minutes.
- **P1 - Critical Degradation**: Primary API endpoints error rate > 5%, webhook deliveries delayed > 15 minutes, or major feature failure without workaround. SLA response: < 30 minutes.
- **P2 - Moderate**: Degraded performance, non-critical subservice down, workaround available. SLA: < 2 hours.
- **P3 - Minor / Cosmetic**: Documentation typo, UI glitch, non-blocking bug. SLA: < 24 hours.

## Triage Protocol for Support Agents
1. Verify system status via internal health dashboard (`https://internal-status.platform.io`) and AWS/GCP region health.
2. If ticket reports broad 5xx errors or cluster connection timeouts, verify if an active incident bridge is open in `#incident-bridge`.
3. If confirmed outage:
   - Tag ticket with `active-incident-<incident_id>`.
   - Send standardized incident macro acknowledgement with link to `status.platform.io`.
   - Suppress redundant SLA escalations while Incident Response team works.
4. Post-resolution: Notify affected customers with post-mortem summary and service credit information within 48 hours.
