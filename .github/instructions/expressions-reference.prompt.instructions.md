---
applyTo: "**/*checks*/**/*.java"
---
# Expressions Utility Reference

Import:
```java
import org.sonar.python.checks.utils.Expressions;
```

Public Static Methods:

- boolean isFalsy(@Nullable Expression expression): Checks if an expression is definitely falsy (None, False literal, zero, empty string/list/tuple/dict); typical use to detect statically false conditions.
- boolean isTruthy(@Nullable Expression expression): Checks if an expression is definitely truthy (True literal, non-zero numeric, non-empty literal); use for static truth-value testing.
- @CheckForNull Expression singleAssignedValue(Name name): Returns the single assigned value for a variable if assigned exactly once; useful to resolve simple variable indirections.
- Expression removeParentheses(Expression expression): Strips enclosing parentheses from an expression; handy to simplify AST node analysis.
- Optional<Expression> singleAssignedNonNameValue(Name name): Resolves a variable to its non-name assigned value, skipping name-only indirections; useful for inlining constant or literal values.
- Optional<Expression> ifNameGetSingleAssignedNonNameValue(Expression expression): If expression is a name, returns its underlying non-name assigned value; otherwise returns the original; for unified value resolution.
- List<Expression> expressionsFromListOrTuple(Expression expression): Extracts elements from a ListLiteral or Tuple node; use to iterate literal collections in code.
- String unescape(StringLiteral stringLiteral): Concatenates and unescapes parts of a string literal into its runtime value; use to get actual string content.
- String unescapeString(String value, boolean isBytesLiteral): Unescapes escape sequences in a raw string value, considering bytes vs string semantics; visible for testing literal unescaping.
- Optional<Name> getAssignedName(Expression expression): Finds the single name to which an expression is assigned, with limited recursion; useful for tracking simple assignments.
- List<Expression> getExpressionsFromRhs(Expression rhs): Gathers expressions from RHS of tuple/list/unpacking assignments; use in multi-variable assignment analysis.
- boolean isGenericTypeAnnotation(Expression expression): Detects calls to typing.TypeVar (generic type annotations); use to identify TypeVar usage in type declarations.
- boolean containsSpreadOperator(List<Argument> arguments): Returns true if any argument uses unpacking (`*` or `**`); use to detect spread operator in call arguments.
