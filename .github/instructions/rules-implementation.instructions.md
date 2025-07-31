---
applyTo: "**/*checks*/**/*.java"
---
Here is some information about basic rule implementation:

- When implementing a new rule, the rule has to be added to [OpenSourceCheckList.java](../../python-checks/src/main/java/org/sonar/python/checks/OpenSourceCheckList.java).
- The class name ends in `Check`, for example `SleepZeroInAsyncCheck`
- Annotate the class with @Rule(key = "YourRuleKey"):
Replace "YourRuleKey" with a unique identifier for your rule (e.g., "S1234").
- Always start out by having the rule extend `PythonSubscriptionCheck` as a first step
- Override the initialize(Context context) method:
    This method is called once when the analysis starts.
    Use context.registerSyntaxNodeConsumer(Kind.YOUR_TARGET_KIND, this::yourCheckMethod) to register a callback method for specific Python Abstract Syntax Tree (AST) node kinds you want to inspect.
    Kind.YOUR_TARGET_KIND refers to the type of Python code structure you're interested in (e.g., Kind.CALL_EXPR for function calls, Kind.FUNCTION_DEF for function definitions, Kind.IF_STMT for if statements, etc.).
    Make sure to use a Kind that exists in the Enum.
this::yourCheckMethod is a method reference to the method in your rule class that will be called when a node of the specified Kind is encountered.
- Implement your check method(s) (e.g., yourCheckMethod(SubscriptionContext context)):
    This method will be called for each AST node of the Kind you registered in initialize.
    Get the current AST node being visited using context.syntaxNode(). You'll likely need to cast it to its specific type (e.g., (CallExpression) context.syntaxNode()).
    Implement the logic to analyze the AST node and its properties.
    You can use TreeUtils for helper functions to navigate or inspect the AST (e.g., TreeUtils.firstAncestorOfKind(...), TreeUtils.nthArgumentOrKeyword(...)).
    If you need to perform type checking:
    You can initialize a TypeCheckMap in a method registered for Kind.FILE_INPUT (as seen with initializeTypeCheckMap in the example).
    Use context.typeChecker().typeCheckBuilder() to define type checks (e.g., checking for a fully qualified name like isTypeWithFqn("module.submodule.function")).
    Always prefer using the type checkers instead fully qualified names on the symbol
- Report issues:
    If your rule's conditions are met and an issue should be raised, use context.addIssue(treeNode, message).
    treeNode is the AST node to which the issue should be attached.
    message is the description of the issue.
    You can add secondary locations to an issue using issue.secondary(anotherTreeNode, secondaryMessage).
- Define message strings:
    Use constants for messages (e.g., private static final String MESSAGE = "Your informative message here.").
    Use String.format() if your messages need to include dynamic parts.
- (Optional) Create helper records or classes:
    For more complex logic or to hold state/configuration related to specific checks (like MessageHolder in the example), you can define inner records or classes.

Commonly Used APIs and Utilities for Rule Implementation
- TreeUtils (org.sonar.python.tree.TreeUtils):
    firstAncestorOfKind(tree, Kind...): Find the first ancestor node of a given kind.
    firstAncestor(tree, Predicate<Tree>): Find the first ancestor matching a predicate.
    hasDescendant(tree, Predicate<Tree>): Check if a tree has a descendant matching a predicate.
    getSymbolFromTree(tree): Get the symbol associated with a tree node, if any.
    getClassSymbolFromDef(classDef), getFunctionSymbolFromDef(functionDef): Get class/function symbol from definition node.
    nthArgumentOrKeyword(int pos, String keyword, List<Argument>): Get the nth or keyword argument from a call. Also has an Optional variant
    argumentByKeyword(String keyword, List<Argument>): Get argument by keyword.
    isBooleanLiteral(tree): Check if a tree is a boolean literal.
    toOptionalInstanceOf(Class<T>, tree), toStreamInstanceOfMapper(Class<T>): Safe casting and mapping for tree nodes. Useful for Optionals and avoiding nulls.
    firstChild(tree, Predicate<Tree>): Find the first child matching a predicate.
- Type Checking:
    TypeCheckMap and TypeCheckBuilder (org.sonar.python.types.v2): Used to map type checks to actions or data, e.g., for matching function calls by type.
    context.typeChecker().typeCheckBuilder().isTypeWithFqn("..."), isTypeOrInstanceWithName("..."), etc.: Build type checks for use in rules.
- SubscriptionContext (org.sonar.plugins.python.api.SubscriptionContext):
    syntaxNode(): Get the current AST node being visited.
    addIssue(tree, message): Report an issue at a given node.
    typeChecker(): Access type checking utilities.
    pythonFile(): Access file-level information.
- Tree.Kind (org.sonar.plugins.python.api.tree.Tree.Kind):
    Used to register node consumers and to check node types (e.g., Kind.CALL_EXPR, Kind.FUNCDEF, Kind.CLASSDEF, etc.).
Patterns:
    Use context.registerSyntaxNodeConsumer(Kind.X, this::method) in initialize to register callbacks for AST node kinds.
    Use type checks and TypeCheckMap for rules that depend on type information.
    Prefer using TypeV2 instead of the V1 symbols
    When initializing TypeCheckBuilder objects, make a separate method.
    In general, the callbacks should be in their own functions and not a lambda
    Make sure to use the correct classes, for exemple most of the Tree objects have names that do not end in Tree. Before using, make sure the types exist.
    When building type checkers, always recreate the builder from the context. Do not reuse one.
    When casting to a tree, use a variable name that makes sense like expr, callExpression, ...