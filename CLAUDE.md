# CLAUDE.md

SonarQube Python Plugin - static analysis for Python code 

## Build

```bash
mvn clean install
```

## Testing

```bash
# Run single test class
mvn test -pl <module> -Dtest=<TestClassName>

# Run single test method
mvn test -pl <module> -Dtest=<TestClassName>#<methodName>
```

Example: `mvn test -pl python-frontend -Dtest=PythonParserTest#test_python`

## Modules

- `python-frontend/` - Parser, AST, semantic model, type inference
- `python-checks/` - Rule implementations (checks)
- `python-checks-testkit/` - Test utilities for checks
- `python-commons/` - Shared utilities
- `sonar-python-plugin/` - Plugin packaging
- `its/` - Integration tests
- `private/python-enterprise-checks/` - Enterprise rules (PySpark, PyTorch)
- `private/sonar-python-enterprise-plugin/` - Enterprise plugin packaging
- `private/its-enterprise/` - Enterprise integration tests & ruling

## Key Paths

- Checks: `python-checks/src/main/java/org/sonar/python/checks/`
- Rule metadata: `python-checks/src/main/resources/org/sonar/l10n/py/rules/python/`
- AST/Tree API: `python-frontend/src/main/java/org/sonar/plugins/python/api/tree/`
- Semantic/Types: `python-frontend/src/main/java/org/sonar/python/semantic/`
- Typeshed stubs generator: `python-frontend/typeshed_serializer/`
