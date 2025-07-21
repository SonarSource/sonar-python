---
description: Use this rule for any rule implementation
applyTo: "**"
---
# TDD with Planning-Execution-Refactoring Methodology

## Overview

This rule establishes a systematic approach for implementing rules and features using Test-Driven Development with explicit Planning, Execution, and Refactoring phases.

## The Three-Phase Approach

### Phase 1: Planning 🎯

**Before writing any code, create a detailed implementation plan:**

1. **Analyze Requirements**
   - Understand the rule specification completely
   - Identify all edge cases and scenarios
   - Review existing test files to understand expected behavior

2. **Design the Architecture**
   - Choose appropriate patterns (e.g., TypeCheckMap for type-based rules)
   - Plan the class structure and method signatures
   - Consider data-driven approaches using static Sets/Lists
   - Write pseudo-code outlining the main logic

3. **Plan Testing Strategy**
   - Identify test cases that need to pass
   - Understand the verification framework being used
   - Plan the test execution command

**Example Planning Output:**
```java
// Plan: Use TypeCheckMap to detect sync subprocess calls in async functions
private final TypeCheckMap<Object> typeChecks = new TypeCheckMap<>();
private static final Set<String> SYNC_CALLS = Set.of("subprocess.run", ...);

// Logic:
// 1. Register FILE_INPUT to init TypeCheckMap
// 2. Register CALL_EXPR to check each call
// 3. Use TreeUtils.asyncTokenOfEnclosingFunction() to detect async context
```

### Phase 2: Execution ⚡

**Implement the planned solution systematically:**

1. **Skeleton First**
   - Create the basic class structure with @Rule annotation
   - Add required imports and extend appropriate base class
   - Implement empty initialize() method

2. **Core Logic Implementation**
   - Follow the planned architecture exactly
   - Use established patterns (see examples below)
   - Implement one piece at a time
   - Test frequently with `mvn verify -Dtest=YourTestClass`

3. **Handle Compilation Issues**
   - Fix import issues immediately
   - Use correct AST node types (e.g., QualifiedExpression not MemberExpression)
   - Verify method signatures match the APIs

4. **Test and Iterate**
   - Run tests after each significant change
   - If tests fail, analyze the output carefully
   - Make targeted fixes (e.g., use `isTypeOrInstanceWithName` vs `isTypeWithFqn`)
   - Ensure license compliance with `mvn license:format`

### Phase 3: Refactoring 🔄

**After tests pass, improve the code quality:**

1. **Code Review and Cleanup**
   - Apply DRY (Don't Repeat Yourself) principle
   - Apply KISS (Keep It Simple, Stupid) principle
   - Apply YAGNI (You Aren't Gonna Need It) principle
   - Remove unnecessary abstractions

2. **Structure Improvements**
   - Consolidate multiple TypeCheckMaps into one if possible
   - Replace inline logic with data-driven approaches
   - Remove helper methods that don't add value
   - Improve naming and organization

3. **Final Verification**
   - Run all tests to ensure refactoring didn't break functionality
   - Verify code follows established patterns in the codebase
   - Ensure clean, maintainable result

## Established Patterns to Follow

### Rule Implementation Pattern
Reference: [SynchronousSubprocessOperationsInAsyncCheck.java](../../python-checks/src/main/java/org/sonar/python/checks/SynchronousSubprocessOperationsInAsyncCheck.java)

```java
@Rule(key = "SXXXX")
public class YourRuleCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Your message here";
  private static final Set<String> TARGET_CALLS = Set.of("call1", "call2");
  private final TypeCheckMap<Object> typeChecks = new TypeCheckMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initTypeChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCall);
  }
}
```

### Testing Pattern
Reference: [SynchronousSubprocessOperationsInAsyncCheckTest.java](../../python-checks/src/test/java/org/sonar/python/checks/SynchronousSubprocessOperationsInAsyncCheckTest.java)

```java
@Test
void test() {
  PythonCheckVerifier.verify("src/test/resources/checks/yourTestFile.py", new YourRuleCheck());
}
```

### Test Execution Command
```bash
mvn verify -DskipTypeshed -DskipObfuscation -f python-checks/pom.xml -Dtest=YourTestClass
```

## Key Success Practices

1. **Data-Driven Design**: Use static Sets/Lists to drive TypeCheckMap population
2. **Single Responsibility**: Each TypeCheckMap should have a clear, single purpose
3. **Early Testing**: Test after each significant implementation step
4. **Proper AST Navigation**: Use TreeUtils helper methods consistently
5. **Type Checking**: Prefer `isTypeOrInstanceWithName` for function calls
6. **Issue Reporting**: Include secondary locations for context (e.g., async keyword)

## Anti-Patterns to Avoid

- ❌ Writing all code before testing
- ❌ Multiple TypeCheckMaps when one suffices
- ❌ Helper methods that just wrap simple operations
- ❌ Using wrong AST node types (check existing code for examples)
- ❌ Skipping the planning phase and coding directly

## Success Metrics

- ✅ All tests pass on first complete implementation
- ✅ Code follows established patterns in the codebase
- ✅ Refactored code is cleaner and more maintainable
- ✅ Implementation matches the original plan closely
- ✅ No compilation or linting errors
