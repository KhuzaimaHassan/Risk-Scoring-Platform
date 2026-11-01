# Risk Scoring Platform

A fraud-risk scoring demo built with FastAPI, PostgreSQL, scikit-learn, and React. It scores stored transactions, records predictions with feature snapshots and model versions, and displays results in an authenticated dashboard.

[Live dashboard](https://risk-scoring-platform-sg.onrender.com/ui/) · [Interactive API docs](https://risk-scoring-platform-sg.onrender.com/docs) · [Service health](https://risk-scoring-platform-sg.onrender.com/api/v1/health)

The demo runs on Render Free in Singapore. It can take a minute or longer to wake up after inactivity. Admin credentials are private; there is no public default login.

## Documentation

Each link opens a dedicated Markdown guide.

| Guide | What it covers |
| --- | --- |
| [Features](docs/FEATURES.md) | Scoring, authentication, audit records, dashboard, and review workflow |
| [Architecture](docs/ARCHITECTURE.md) | System flow, components, database, and serving registry |
| [Local setup](docs/SETUP.md) | Prerequisites, environment, disposable test data, and running locally |
| [Deployment](docs/DEPLOYMENT.md) | Render configuration, persistence, free-tier limits, and troubleshooting |
| [API reference](docs/API.md) | Routes, authentication, request/response contracts, timings, and errors |
| [Frontend development](docs/FRONTEND.md) | Vite development, build commands, and dashboard behavior |
| [Verification](docs/VERIFICATION.md) | Tests, CI, PostgreSQL integration, and API smoke checks |
| [Models and training](docs/MODELS.md) | Candidate training, promotion, monitoring, and model lifecycle |
| [Performance and limitations](docs/PERFORMANCE.md) | Measured latency, model metrics, and remaining demo limitations |

This is an educational demo. No license file is included, and no open-source license grant is asserted.
