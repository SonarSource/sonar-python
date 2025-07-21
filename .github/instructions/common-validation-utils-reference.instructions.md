---
description: ""
applyTo: "**/*checks*/**/*.java"
---
# CommonValidationUtils Reference

Import:
```java
import org.sonar.python.checks.hotspots.CommonValidationUtils;
```

## Public Static Methods

- boolean isMoreThan(Expression expression, int number) : Checks if an expression's numeric value is strictly greater than the given threshold, resolving name indirections; typical use: detecting sleep durations exceeding a limit (e.g., in `AsyncLongSleepCheck`).

- boolean isEqualTo(Expression expression, int number) : Determines if an expression represents a numeric literal equal to a specified value (integer or double), resolving name indirections; typical use: validating exact numeric matches in code analysis.
