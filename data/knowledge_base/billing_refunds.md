# Billing, Invoices, and Refund Policy

## Overview
This runbook covers customer billing inquiries, disputed charges, invoice retrieval, and refund evaluation.

## Refund Eligibility
- **14-Day Money-Back Guarantee**: Full refund eligible within 14 days of the initial subscription charge or renewal for annual plans.
- **Service Outage Credit**: Customers affected by verified P0/P1 platform outages exceeding our 99.9% uptime SLA are eligible for service credits calculated pro-rata on their monthly invoice.
- **Prorated Upgrades/Downgrades**: Plan upgrades take effect immediately with prorated billing. Downgrades take effect at the end of the current billing cycle.

## Common Error Codes
- `ERR_PAYMENT_FAILED_01`: Insufficient funds or card expired. Guide user to `Settings > Billing > Payment Methods` to update details.
- `ERR_CURRENCY_MISMATCH`: Enterprise invoices must match the registered tax jurisdiction currency.
- `ERR_INVOICE_LOCKED`: Invoices older than 90 days cannot be edited directly; contact Finance for adjustments.

## Resolution Steps for Support Agents
1. Verify customer identity and account email via CRM.
2. For refund requests within 14 days: execute 1-click refund via Stripe/Billing connector.
3. For enterprise disputes > $500: escalate to Finance & Billing Operations queue (`finance-tier2`).
4. Re-issue updated VAT/Tax invoices upon request through the customer portal.
