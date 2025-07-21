---
applyTo: "**/*checks*/**/*.java"
---
# TreeUtils Reference

**Import**
```java
import org.sonar.python.tree.TreeUtils;
```

## Public Static Methods

- `Tree firstAncestor(Tree tree, Predicate<Tree> predicate)`
  **Return:** `Tree`
  **Description:** Finds the first ancestor matching the predicate or returns null. Commonly used to locate enclosing scopes like functions, classes, or statement lists.

- `Tree firstAncestorOfKind(Tree tree, Kind... kinds)`
  **Return:** `Tree`
  **Description:** Shortcut to find the first ancestor of any specified kinds. Handy for finding loops, conditionals, class or function definitions.

- `Collector<Tree, ?, Map<Tree, Tree>> groupAssignmentByParentStatementList()`
  **Return:** `Collector<Tree, ?, Map<Tree, Tree>>`
  **Description:** Groups AST nodes by their nearest parent statement list, picking the first occurrence in each block. Used in checks like DeadStoreCheck and IgnoredParameterCheck.

- `Comparator<Tree> getTreeByPositionComparator()`
  **Return:** `Comparator<Tree>`
  **Description:** Comparator ordering trees by source position (line then column). Useful to sort nodes in source order.

- `List<Token> tokens(Tree tree)`
  **Return:** `List<Token>`
  **Description:** Flattens an AST subtree into a list of all tokens (including whitespace). Employed in token-based analyses and quick fixes.

- `List<Token> nonWhitespaceTokens(Tree tree)`
  **Return:** `List<Token>`
  **Description:** Filters out indent, dedent, and newline tokens from `tokens(tree)`. Useful for checks that ignore formatting tokens.

- `boolean hasDescendant(Tree tree, Predicate<Tree> predicate)`
  **Return:** `boolean`
  **Description:** Checks if any descendant of the tree matches the predicate. Handy for detecting nested constructs, e.g., calls inside expressions.

- `Stream<Expression> flattenTuples(Expression expression)`
  **Return:** `Stream<Expression>`
  **Description:** Recursively flattens nested tuple expressions into a stream of leaf expressions. Common in handling exception lists in `except` clauses.

- `Optional<Symbol> getSymbolFromTree(Tree tree)`
  **Return:** `Optional<Symbol>`
  **Description:** Retrieves the symbol associated with a tree node if present. Used for resolving references or type checks.

- `ClassSymbol getClassSymbolFromDef(@Nullable ClassDef classDef)`
  **Return:** `ClassSymbol` or null
  **Description:** Returns the `ClassSymbol` of a class definition or null if none. Throws if symbol kind is unexpected. Used in class-based rules.

- `List<String> getParentClassesFQN(ClassDef classDef)`
  **Return:** `List<String>`
  **Description:** Returns fully qualified names of all ancestor classes. Useful to detect inheritance patterns and mixins.

- `FunctionSymbol getFunctionSymbolFromDef(@Nullable FunctionDef functionDef)`
  **Return:** `FunctionSymbol` or null
  **Description:** Retrieves the `FunctionSymbol` of a function definition or null. Throws if symbol kind mismatch. Employed in method analyses.

- `List<Parameter> nonTupleParameters(FunctionDef functionDef)`
  **Return:** `List<Parameter>`
  **Description:** Returns parameters excluding tuple-unpacked ones. Useful when checks should ignore tuple parameters.

- `List<Parameter> positionalParameters(FunctionDef functionDef)`
  **Return:** `List<Parameter>`
  **Description:** Collects positional parameters up to the first `*`. Handy for analyzing positional-only arguments.

- `List<FunctionDef> topLevelFunctionDefs(ClassDef classDef)`
  **Return:** `List<FunctionDef>`
  **Description:** Collects top-level function definitions inside a class, ignoring nested functions. Used for method discovery in conditional blocks.

- `int findIndentationSize(Tree tree)`
  **Return:** `int`
  **Description:** Computes indentation size (spaces) between a node and its parent or within subtree. Useful for quick-fix indentation alignment.

- `RegularArgument argumentByKeyword(String keyword, List<Argument> arguments)`
  **Return:** `RegularArgument` or null
  **Description:** Finds a call argument by keyword name. Returns null if absent. Common in argument presence or value checks.

- `RegularArgument nthArgumentOrKeyword(int argPosition, String keyword, List<Argument> arguments)`
  **Return:** `RegularArgument` or null
  **Description:** Retrieves the nth positional or matching keyword argument. Use to inspect specific call parameters.

- `Optional<RegularArgument> nthArgumentOrKeywordOptional(int argPosition, String keyword, List<Argument> arguments)`
  **Return:** `Optional<RegularArgument>`
  **Description:** Same as `nthArgumentOrKeyword` but wrapped in `Optional` for safe chaining.

- `boolean isBooleanLiteral(Tree tree)`
  **Return:** `boolean`
  **Description:** Checks if a name node represents Python `True` or `False`. Useful in boolean literal rules.

- `String nameFromExpression(Expression expression)`
  **Return:** `String` or null
  **Description:** Returns the identifier name if the expression is a simple `Name`, otherwise null. Used to extract variable names.

- `Optional<String> nameFromQualifiedOrCallExpression(Expression expression)`
  **Return:** `Optional<String>`
  **Description:** Extracts name from a qualified or call expression. Handy when decorators or calls appear.

- `Optional<String> nameFromExpressionOrQualifiedExpression(Expression expression)`
  **Return:** `Optional<String>`
  **Description:** Returns name from `Name` or `QualifiedExpression`. Use in identifier resolution.

- `String nameFromQualifiedExpression(QualifiedExpression qualifiedExpression)`
  **Return:** `String`
  **Description:** Builds a dotted name string from a `QualifiedExpression`. Useful for decorator or attribute access checks.

- `String decoratorNameFromExpression(Expression expression)`
  **Return:** `String` or null
  **Description:** Extracts decorator name from an expression, handling calls and qualifiers. Used in decorator-based rules.

- `Optional<Token> asyncTokenOfEnclosingFunction(Tree tree)`
  **Return:** `Optional<Token>`
  **Description:** Retrieves the `async` keyword token of the enclosing function, if any. Useful for async rules.

- `boolean isFunctionWithGivenDecoratorFQN(Tree tree, String decoratorFQN)`
  **Return:** `boolean`
  **Description:** Checks if a function definition has a decorator with the specified fully qualified name.

- `boolean isDecoratorWithFQN(Decorator decorator, String fullyQualifiedName)`
  **Return:** `boolean`
  **Description:** Checks if a decorator matches the given fully qualified name. Used to identify specific decorators.

- `Optional<String> fullyQualifiedNameFromQualifiedExpression(QualifiedExpression qualifiedExpression)`
  **Return:** `Optional<String>`
  **Description:** Builds fully qualified name from a qualified expression. Useful in semantic analysis.

- `Optional<String> fullyQualifiedNameFromExpression(Expression expression)`
  **Return:** `Optional<String>`
  **Description:** Derives FQN from name, qualified, or call expressions. Aids in type-based and decorator rules.

- `LocationInFile locationInFile(Tree tree, @Nullable String fileId)`
  **Return:** `LocationInFile` or null
  **Description:** Constructs a `LocationInFile` mapping a tree node to file offsets. Used in issue reporting and quick fixes.

- `Token getTreeSeparatorOrLastToken(Tree tree)`
  **Return:** `Token`
  **Description:** Returns statement separator (e.g., semicolon) or last token. Useful for handling token ranges.

- `<T> Function<Tree, T> toInstanceOfMapper(Class<T> castToClass)`
  **Return:** `Function<Tree, T>`
  **Description:** Returns a mapper that casts a tree to a given class or returns null. Handy in stream pipelines.

- `<T> Function<Tree, Optional<T>> toOptionalInstanceOfMapper(Class<T> castToClass)`
  **Return:** `Function<Tree, Optional<T>>`
  **Description:** Returns a mapper to cast a tree to an Optional of the given class.

- `<T> Optional<T> toOptionalInstanceOf(Class<T> castToClass, @Nullable Tree tree)`
  **Return:** `Optional<T>`
  **Description:** Safely casts a tree to the given class and wraps in Optional.

- `<T> Function<Tree, Stream<T>> toStreamInstanceOfMapper(Class<T> castToClass)`
  **Return:** `Function<Tree, Stream<T>>`
  **Description:** Returns a mapper producing a stream of instances of the given class from a tree node.

- `Optional<Tree> firstChild(Tree tree, Predicate<Tree> filter)`
  **Return:** `Optional<Tree>`
  **Description:** Finds the first child in the subtree matching the filter predicate. Used to locate specific sub-nodes.

- `String treeToString(Tree tree, boolean renderMultiline)`
  **Return:** `String` or null
  **Description:** Renders tree tokens back to source string, optionally only single-line. Useful for code generation in quick fixes.

- `List<String> dottedNameToPartFqn(DottedName dottedName)`
  **Return:** `List<String>`
  **Description:** Splits a dotted name into its part FQNs. Handy when handling `DottedName` nodes.
