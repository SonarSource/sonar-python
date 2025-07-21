---
description: This rule should be attached anytime there is a quickfix involved.
applyTo: "**"
---
# Python Plugin Quick-Fix Creation Guidelines

This rule codifies the patterns and best practices for creating quick-fixes (code transformations) in the SonarQube Python plugin codebase.

## 1. Use the `PythonQuickFix` Builder API
- Always start with `PythonQuickFix.newQuickFix(description)` to obtain a `Builder`.
- Example:
  ```java
  var quickFix = PythonQuickFix.newQuickFix("Replace with `math.isclose()`");
  ```
  ([FloatingPointEqualityCheck.java](../../python-checks/src/main/java/org/sonar/python/checks/FloatingPointEqualityCheck.java))

## 2. Imperative, concise descriptions
- Descriptions must:
  - Begin with a verb (`Remove`, `Replace`, `Add`, `Make`).
  - Omit trailing punctuation.
  - Be defined as a `static final String` or via `String.format(...)` for dynamic parts.
- Bad: `"Please remove trailing whitespace."`
- Good: `"Remove trailing whitespace"`
  ([TrailingWhitespaceCheck.java](../../python-checks/src/main/java/org/sonar/python/checks/TrailingWhitespaceCheck.java))

## 3. Leverage `TextEditUtils` helpers
- Do not manually compute offsets; use `TextEditUtils` (e.g. `replace()`, `removeUntil()`, `insertBefore()`, `renameAllUsages()`) to construct `PythonTextEdit` instances.
- Example:
  ```java
  quickFix.addTextEdit(TextEditUtils.replace(be, replacement));
  ```
  ([NumpyIsNanCheck.java](../../python-checks/src/main/java/org/sonar/python/checks/NumpyIsNanCheck.java))

## 4. Accumulate multiple edits when needed
- Chain `.addTextEdit(...)` calls on the `Builder` to perform multi-step transformations (e.g., insert imports, replace code snippets).
- Always finalize with `.build()`.

## 5. Offer fixes conditionally
- Attach a quick-fix only when the transformation is guaranteed safe and complete.
- Use `Optional<PythonQuickFix>` or guard logic to only add fixes in simple, deterministic cases.

## 6. Testing and Verification

- Replace the existing section with detailed instructions on writing and running quick-fix tests.

### 6.1 Writing quick-fix tests
- Write tests with:
  ```java
  PythonQuickFixVerifier.verify(
    new MyCheck(),         // your check instance
    "<code before quick-fix>",
    "<code after quick-fix>"
  );
  ```
- For IPython-specific scenarios, use:
  ```java
  PythonQuickFixVerifier.verifyIPython(...);
  PythonQuickFixVerifier.verifyIPythonQuickFixMessages(...);
  ```
- To assert expected failures, leverage AssertJ:
  ```java
  assertThatThrownBy(() -> PythonQuickFixVerifier.verify(...))
    .isInstanceOf(AssertionError.class)
    .hasMessageContaining("[Number of quickfixes]");
  ```

## 7. Reference to TextEditUtils
For best practices on using text editing utilities, refer to the [TextEditUtils Reference](./text-edit-utils-reference.instructions.md). This resource provides detailed information on the available methods for manipulating code, which can enhance the effectiveness of your quick-fix implementations.

---

Adhering to these guidelines will keep quick-fix implementations consistent, reliable, and maintainable across the Python plugin.
