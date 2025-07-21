---
description: "Reference for CheckUtils utility methods."
applyTo: "**/*checks*/**/*.java"
---
# CheckUtils Reference

Import :
```java
import org.sonar.python.checks.utils.CheckUtils;
```

## Public Static Methods

- boolean areEquivalent(Tree leftTree, Tree rightTree) : Compares two AST subtrees for structural and token-value equivalence; useful in rules or tests to detect duplicate or matching nodes.
- ClassDef getParentClassDef(Tree tree) : Finds the nearest enclosing class definition or returns null if none; used to determine the class context for a given AST node.
- boolean classHasInheritance(ClassDef classDef) : Checks if a class declares any base other than the default `object`; used to detect or enforce explicit inheritance patterns.
- boolean containsCallToLocalsFunction(Tree tree) : Determines if a subtree contains any call to `locals()`; helpful in rules detecting dynamic locals usage.
- boolean isConstant(Expression condition) : Checks if an expression is an immutable literal or a literal collection without unpacking; used to identify compile-time constant conditions in checks.
- boolean isImmutableConstant(Expression condition) : Returns true for boolean, numeric, string literals, `None`, lambdas, or generator expressions; useful for primitive constant detection.
- boolean isConstantCollectionLiteral(Expression condition) : Returns true for list, dict, set, or tuple literals without unpacking; used to detect static collection declarations in code.
- boolean mustBeAProtocolLike(ClassDef classDef) : Determines if a class extends a `typing.Protocol` or similar interface; used in protocol-specific or interface checks.
- boolean isAbstract(FunctionDef funDef) : Checks whether a function is decorated as an abstract method; helpful in analyzing abstract base classes.
- boolean isEmptyStatement(Statement statement) : Identifies `pass` statements, standalone string literals, or ellipsis as empty/no-op statements; used to skip over no-op code.
- boolean isSelf(Expression expression) : Checks if an expression is the name `self`; useful to special-case instance references in method checks.
- Symbol findFirstParameterSymbol(FunctionDef functionDef) : Retrieves the symbol of the first parameter or returns null; used in parameter-name based analyses and naming rules.
