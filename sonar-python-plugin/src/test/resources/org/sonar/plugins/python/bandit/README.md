# Bandit Report Generation

```
(
  cd sonar-python-plugin/src/test/resources/org/sonar/plugins/python && \
  bandit --format json -o bandit/bandit-report.json -r bandit
)
(
  cd its/plugin/projects/bandit_project && \
  bandit --format json -o bandit-report.json -r src
)
```
