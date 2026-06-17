# Python Framework Usage Among Users

Date: 2026-06-17

## Summary

This note ranks the most used Python frameworks and adjacent tooling among external SonarCloud users, based on detected `PYPI` dependencies in active projects.

Population:

- 169 external organizations
- 1,861 active projects with PYPI dependencies

Key takeaway:

- If all Python tooling is included, the most prevalent exact packages are Requests, PyYAML, Cryptography, Click, Jinja2, NumPy, pytest, and Pydantic.
- If the scope is narrowed to web frameworks and serving stack, FastAPI leads the framework packages visible in this dataset, followed by aiohttp and Flask, with Uvicorn the most common server package.
- For testing, pytest is the clear leader, followed by Coverage.py and pytest-cov.

## Ranked Table

| family | category | orgs | projects | org_pct | project_pct |
| --- | --- | ---: | ---: | ---: | ---: |
| Requests | http client | 136 | 1349 | 80.5 | 72.5 |
| PyYAML | config/data serialization | 115 | 1041 | 68.0 | 55.9 |
| Cryptography | security/crypto | 110 | 696 | 65.1 | 37.4 |
| Click | cli framework | 109 | 957 | 64.5 | 51.4 |
| Jinja2 | templating | 109 | 800 | 64.5 | 43.0 |
| NumPy | data/scientific computing | 107 | 664 | 63.3 | 35.7 |
| pytest | test framework | 106 | 1045 | 62.7 | 56.2 |
| Pydantic | data validation/settings | 102 | 662 | 60.4 | 35.6 |
| PyJWT | auth/jwt | 98 | 508 | 58.0 | 27.3 |
| Pandas | data analysis | 96 | 622 | 56.8 | 33.4 |
| HTTPX | http client | 87 | 470 | 51.5 | 25.3 |
| Uvicorn | asgi server | 85 | 361 | 50.3 | 19.4 |
| Coverage.py | test coverage | 84 | 844 | 49.7 | 45.4 |
| FastAPI | web framework | 79 | 386 | 46.7 | 20.7 |
| pytest-cov | test coverage plugin | 77 | 786 | 45.6 | 42.2 |
| aiohttp | async web/http framework | 74 | 371 | 43.8 | 19.9 |
| SQLAlchemy | orm/database toolkit | 73 | 379 | 43.2 | 20.4 |
| Boto3 | aws sdk | 71 | 732 | 42.0 | 39.3 |
| Flask | web framework | 69 | 189 | 40.8 | 10.2 |
| SciPy | scientific computing | 66 | 210 | 39.1 | 11.3 |
| PyArrow | data/columnar | 65 | 351 | 38.5 | 18.9 |
| OpenTelemetry API | observability | 61 | 296 | 36.1 | 15.9 |
| Beautiful Soup | html parsing | 56 | 394 | 33.1 | 21.2 |
| Ruff | linting/formatting | 55 | 434 | 32.5 | 23.3 |
| pytest-asyncio | test plugin | 54 | 217 | 32.0 | 11.7 |
| Gunicorn | wsgi server | 52 | 155 | 30.8 | 8.3 |
| scikit-learn | machine learning | 52 | 162 | 30.8 | 8.7 |
| Typer | cli framework | 51 | 197 | 30.2 | 10.6 |
| pytest-mock | test plugin | 50 | 234 | 29.6 | 12.6 |
| Matplotlib | plotting | 48 | 159 | 28.4 | 8.5 |
| Black | formatter | 47 | 431 | 27.8 | 23.2 |
| isort | formatter/import sorting | 47 | 320 | 27.8 | 17.2 |

## By Interpretation

### Web frameworks and serving stack

| family | orgs | projects | org_pct |
| --- | ---: | ---: | ---: |
| Uvicorn | 85 | 361 | 50.3 |
| FastAPI | 79 | 386 | 46.7 |
| aiohttp | 74 | 371 | 43.8 |
| Flask | 69 | 189 | 40.8 |
| Gunicorn | 52 | 155 | 30.8 |

### Test frameworks and plugins

| family | orgs | projects | org_pct |
| --- | ---: | ---: | ---: |
| pytest | 106 | 1045 | 62.7 |
| Coverage.py | 84 | 844 | 49.7 |
| pytest-cov | 77 | 786 | 45.6 |
| pytest-asyncio | 54 | 217 | 32.0 |
| pytest-mock | 50 | 234 | 29.6 |

## Method

Relevant assets were identified through Atlan, then queried through Sonardata.

Primary warehouse tables:

- `sonar.fct_sc_sca_release`
- `sonar.fct_sc_project`

Population and join:

- Filtered to `package_manager = 'PYPI'`
- Filtered to external orgs with `is_internal = false`
- Filtered to active projects with `is_active_project = true`
- Joined `sonar.fct_sc_sca_release.component_uuid = sonar.fct_sc_project.project_uuid_v4`

Aggregation logic:

- Counted distinct organizations via `organization_uuid_v4`
- Counted distinct projects via `project_uuid_v4`
- Ranked exact Python packages detected in the dependency data

## Caveats

- This is dependency-detection based, not `requirements.txt` or `pyproject.toml` intent based.
- Python package usage is less namespace-structured than JS/TS framework ecosystems, so this export uses exact package rankings rather than merged broad families.
- The table intentionally mixes frameworks, servers, test packages, SDKs, and common libraries because that is what the dependency data exposes most reliably.
- Some frameworks may appear lower than expected if they are not present in the top exact-package set for this population.

## File

Raw export: [python_framework_usage_2026-06-17.csv](/Users/romain.birling/python_framework_usage_2026-06-17.csv)
