---
name: type-matchers
description: Guide for using the TypeMatchers API in the SonarQube Python Plugin. Use this skill when working with type matching, type checking, or implementing rules that need to verify Python types. This API replaces deprecated TypeCheckBuilder and V1 symbol/type APIs.
---

# TypeMatchers API Skill

## Overview

The TypeMatchers API is the **strongly preferred** approach for type matching in the SonarQube Python Plugin. It provides a fluent, composable API for checking types in the type inference v2 system.

**Entry Point:** `org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers`

## DEPRECATED APIs - DO NOT USE

The following APIs are **deprecated** and should **NOT** be used in new code:

- ❌ `TypeCheckBuilder` - Old type checking API
- ❌ `symbol()` / `symbolV1()` - Version 1 symbol API
- ❌ `type()` / `typeV1()` - Version 1 type API

**Always use TypeMatchers instead.**

## Core Principles

### 1. Prefer Type Equality Over FQN Matching

**IMPORTANT:** Avoid matching on Fully Qualified Names (FQNs) when possible.

**Why?**
- Stubs can be relocated
- Types can be accessed through re-exports
- FQN matching will fail to recognize types in these cases

**Bad Example:**
```java
TypeMatchers.withFQN("typing.Dict")  // ❌ May fail with re-exports
```

**Good Example:**
```java
TypeMatchers.isType("typing.Dict")  // ✅ Uses type table resolution + equality
```

**How it works:** `isType(fqn)` first resolves the FQN through the project's type table, then checks equality between the resolved type and the type on the given expression. If a matcher doesn't explicitly mention "FQN" in its name, it uses type equality.

### 2. ObjectType Handling

TypeMatchers are **strict** about ObjectTypes (instances of class types). They do **NOT** automatically unwrap object types.

**Important:** Use `isObjectSatisfying()` to unwrap ObjectTypes.

**Example:**
```java
// If you want to check if an expression is an instance of "typing.Dict"
TypeMatchers.isObjectOfType("typing.Dict")  // ✅ Composite helper
// Equivalent to:
TypeMatchers.isObjectSatisfying(TypeMatchers.isType("typing.Dict"))
```

### 3. UnionType Handling

UnionTypes are handled **transparently**. For a matcher to return `true` for a UnionType, **ALL** candidates of the union must match.

**Example:**
```java
// For type: Union[int, str]
// Matcher only returns true if BOTH int AND str match the predicate
```

## Available TypeMatchers

### Logical Combinators

#### `all(TypeMatcher...)`
Combines multiple matchers with AND logic. All matchers must match.

```java
TypeMatchers.all(
  TypeMatchers.isType("builtins.dict"),
  TypeMatchers.hasMember("keys")
)
```

Overloads: `all(Stream<TypeMatcher>)`, `all(List<TypeMatcher>)`

#### `any(TypeMatcher...)`
Combines multiple matchers with OR logic. At least one matcher must match.

```java
TypeMatchers.any(
  TypeMatchers.isType("builtins.list"),
  TypeMatchers.isType("builtins.tuple")
)
```

Overloads: `any(Stream<TypeMatcher>)`, `any(List<TypeMatcher>)`

### Basic Type Matching

#### `isType(String fqn)`
Checks if the type equals the type resolved from the given FQN. **Preferred over `withFQN()`**.

```java
TypeMatchers.isType("typing.Dict")
```

#### `withFQN(String fqn)` ⚠️
Matches on FQN directly. **Discouraged** - use `isType()` instead.

```java
TypeMatchers.withFQN("typing.Dict")  // ⚠️ Avoid when possible
```

### Object Type Matching

#### `isObjectSatisfying(TypeMatcher matcher)`
Unwraps an ObjectType and applies the given matcher to the wrapped type.

```java
TypeMatchers.isObjectSatisfying(TypeMatchers.isType("builtins.str"))
```

#### `isObjectOfType(String fqn)`
Convenience method combining `isObjectSatisfying()` and `isType()`.

```java
TypeMatchers.isObjectOfType("builtins.str")
// Equivalent to:
// TypeMatchers.isObjectSatisfying(TypeMatchers.isType("builtins.str"))
```

### Subtype Matching

#### `isSubtypeOf(String fqn)`
Checks if the type is a subtype of the type resolved from the FQN.

```java
TypeMatchers.isSubtypeOf("typing.Mapping")
```

#### `isObjectOfSubType(String fqn)`
Convenience method for object instances of subtypes.

```java
TypeMatchers.isObjectOfSubType("typing.Mapping")
// Equivalent to:
// TypeMatchers.isObjectSatisfying(TypeMatchers.isSubtypeOf("typing.Mapping"))
```

### Type Hierarchy Matching

#### `isOrExtendsType(String fqn)`
Checks if the type or any of its supertypes equals the type resolved from the FQN.

```java
TypeMatchers.isOrExtendsType("builtins.Exception")
```

#### `isTypeOrSuperTypeSatisfying(TypeMatcher matcher)`
Checks if the type or any of its supertypes satisfies the given matcher.

```java
TypeMatchers.isTypeOrSuperTypeSatisfying(
  TypeMatchers.isType("collections.abc.Mapping")
)
```

#### `isTypeOrSuperTypeWithFQN(String fqn)` ⚠️
Checks type hierarchy using FQN. **Prefer `isOrExtendsType()`** instead.

```java
TypeMatchers.isTypeOrSuperTypeWithFQN("builtins.dict")  // ⚠️ Avoid
```

### Function Owner Matching

#### `isFunctionOwnerSatisfying(TypeMatcher matcher)`
Checks if the function's owner type satisfies the given matcher. Useful for checking method owners.

```java
TypeMatchers.isFunctionOwnerSatisfying(
  TypeMatchers.isType("flask.Flask")
)
```

### Member Matching

#### `hasMember(String memberName)`
Checks if the type has a member with the given name.

```java
TypeMatchers.hasMember("append")
```

#### `hasMemberSatisfying(String memberName, TypeMatcher matcher)`
Checks if the type has a member with the given name that satisfies the matcher.

```java
TypeMatchers.hasMemberSatisfying("append",
  TypeMatchers.isType("types.FunctionType")
)
```

### Type Source Matching

#### `hasTypeSource(TypeSource typeSource)`
Checks if the type has a specific TypeSource (e.g., EXACT, INFERRED, STUBBED).

```java
TypeMatchers.hasTypeSource(TypeSource.EXACT)
```

## Common Patterns

### Checking for List or Tuple of Specific Type

```java
TypeMatchers.any(
  TypeMatchers.isObjectOfType("builtins.list"),
  TypeMatchers.isObjectOfType("builtins.tuple")
)
```

### Checking for Dict-like Types

```java
TypeMatchers.isObjectSatisfying(
  TypeMatchers.isSubtypeOf("typing.Mapping")
)
```

### Checking Method Owner

```java
// Check if a method belongs to a Flask app
TypeMatchers.isFunctionOwnerSatisfying(
  TypeMatchers.isOrExtendsType("flask.Flask")
)
```

### Complex Composition

```java
TypeMatchers.all(
  TypeMatchers.isObjectOfType("builtins.dict"),
  TypeMatchers.hasMember("get"),
  TypeMatchers.hasMember("keys")
)
```

## TypeMatcher Interface

The `TypeMatcher` interface provides two methods for evaluating expressions:

### `evaluateFor(Expression expr, SubscriptionContext ctx)`
Returns a `TriBool` (TRUE, FALSE, or UNKNOWN) indicating whether the expression matches.

```java
TriBool result = matcher.evaluateFor(expression, ctx);
if (result == TriBool.TRUE) {
  // Definitely matches
} else if (result == TriBool.UNKNOWN) {
  // Cannot determine
}
```

### `isTrueFor(Expression expr, SubscriptionContext ctx)`
Returns a boolean - `true` only if the match is definitely true (not unknown).

```java
if (matcher.isTrueFor(expression, ctx)) {
  // Definitely matches
}
```

**Note:** `isTrueFor()` is equivalent to `evaluateFor() == TriBool.TRUE`

## Usage in Rules

When implementing rules, use TypeMatchers to check expression types:

```java
private static final TypeMatcher DICT_MATCHER =
  TypeMatchers.isObjectOfType("builtins.dict");

@Override
public void initialize(Context context) {
  context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Expression callee = callExpression.callee();

    if (DICT_MATCHER.isTrueFor(callee, ctx)) {
      ctx.addIssue(callee, "Issue message here");
    }
  });
}
```

**Example with multiple conditions:**

```java
private static final TypeMatcher FLASK_APP_MATCHER =
  TypeMatchers.isObjectOfType("flask.Flask");

private static final TypeMatcher FLASK_APP_METHOD_MATCHER =
  TypeMatchers.isFunctionOwnerSatisfying(FLASK_APP_MATCHER);

@Override
public void initialize(Context context) {
  context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();

    if (FLASK_APP_METHOD_MATCHER.isTrueFor(functionDef, ctx)) {
      // ...
    }
  });
}
```

## Migration from Old APIs

### TypeCheckBuilder → TypeMatchers

**Before:**
```java
TypeCheckBuilder.isBuiltinWithName("dict")
```

**After:**
```java
TypeMatchers.isObjectOfType("builtins.dict")
```

### Symbol/Type V1 → TypeMatchers

**Before:**
```java
expression.type().canBeOrExtend("typing.Dict")
```

**After:**
```java
TypeMatcher matcher = TypeMatchers.isObjectSatisfying(
  TypeMatchers.isOrExtendsType("typing.Dict")
);
if (matcher.isTrueFor(expression, ctx)) {
  // Handle the match
}
```

## Best Practices

1. ✅ **DO** use `isType()` instead of `withFQN()`
2. ✅ **DO** use `isObjectSatisfying()` or `isObjectOfType()` for instance checks
3. ✅ **DO** cache TypeMatcher instances as static final fields
4. ✅ **DO** use composite matchers like `all()` and `any()` for complex conditions
5. ❌ **DON'T** use deprecated TypeCheckBuilder API
6. ❌ **DON'T** use symbolV1() or typeV1() methods
7. ❌ **DON'T** match on FQN when type equality is possible
8. ❌ **DON'T** forget to unwrap ObjectTypes with `isObjectSatisfying()`

## Summary

The TypeMatchers API provides a robust, composable way to match types in the Python plugin. Always prefer it over deprecated APIs, use type equality over FQN matching, and remember to explicitly handle ObjectTypes with `isObjectSatisfying()`.
