# Security and regulated domains (High risk / Strict lane)

- Threat-model the change. Never hard-code or log secrets. Least privilege for service accounts, DB roles, tokens, IAM.
- Mask PII/PHI in logs, errors and fixtures. Keep an audit trail (who, what, when, which record). Respect data residency.
- Map to the applicable frameworks (e.g. PCI-DSS, SOX, GDPR/DPDP, RBI/IRDAI guidelines) and flag what compliance must confirm. Do not claim compliance.
- Front-end: WCAG 2.1/2.2 AA can be an obligation; consent banners and disclosures are legal text.
