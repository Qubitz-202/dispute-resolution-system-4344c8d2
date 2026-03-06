# dispute-resolution-system

AI-powered dispute and chargeback management system for American Express that automatically processes millions of monthly transaction disputes across 114 million cards globally. The system analyzes dispute claims, retrieves transaction evidence from the real-time data lake processing $1.2 trillion in annual transactions, assesses validity based on historical chargeback patterns, and generates resolution recommendations. Straightforward disputes are resolved autonomously within minutes, while complex cases are routed to human fraud management specialists with pre-compiled evidence packages, reducing resolution time by 70% and improving consistency in decision-making across the global operations footprint. Integrates with American Express's NVIDIA-powered AI infrastructure to maintain sub-100 millisecond processing latency while handling enterprise-scale dispute volumes.

## Deployment

This project deploys automatically via GitHub Actions when code is pushed to `main`.

### Required GitHub Secrets
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

### Manual Deploy
```bash
cd cdk && npm ci && npm run build
cdk deploy --all --require-approval never
```

## Project Structure
- `agent/` — Agent runtime code (generated from base template)
- `cdk/` — CDK infrastructure (TypeScript)
- `.github/workflows/` — CI/CD pipeline
