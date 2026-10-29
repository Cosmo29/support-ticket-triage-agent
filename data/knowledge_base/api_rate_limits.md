# API Rate Limits, HTTP 429 Errors, and Webhook Failures

## Overview
Technical reference for API throughput limits, throttling policies, header diagnostics, and webhook delivery retry schedules.

## Rate Limit Tiers
- **Free Tier**: 60 requests / minute (burst capacity: 10 requests / sec).
- **Scale Tier**: 600 requests / minute (burst capacity: 50 requests / sec).
- **Enterprise Tier**: 5,000 requests / minute (custom burst up to 500 requests / sec upon request).

## Response Headers & Diagnostics
When rate limited, the API returns `HTTP 429 Too Many Requests` with:
- `X-RateLimit-Limit`: Maximum requests allowed in current time window.
- `X-RateLimit-Remaining`: Remaining calls available in window.
- `X-RateLimit-Reset`: Unix timestamp when the quota resets.
- `Retry-After`: Number of seconds the client must pause before retrying.

## Recommended Remediation Steps
1. **Exponential Backoff with Jitter**: Clients should implement retry loops starting at 1s, doubling with random jitter (e.g., `1s + rand(0, 500ms)`, `2s + rand(...)`).
2. **Bulk / Batch Endpoints**: Use `/api/v2/batch` instead of single resource GET/POST loops.
3. **Webhook Subscriptions**: Instead of high-frequency polling, subscribe to real-time Webhook events (`ticket.created`, `ticket.updated`).
4. **Temporary Limit Increase**: Enterprise support can grant a 48-hour temporary quota increase for scheduled data migrations.
