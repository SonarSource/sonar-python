/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.AwsLambdaChecksUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7613")
public class AwsLambdaReturnValueAreSerializableCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Fix the return value to be JSON serializable.";
  private static final String SECONDARY_LOCATION_MESSAGE = "The non-serializable value is set here.";

  // This should be moved to type checking once SONARPY-3223 is done
  private static final Set<String> NON_SERIALIZABLE_FQNS = Set.of(
    "datetime.datetime.now",
    "datetime.datetime.utcnow",
    "datetime.datetime.today",
    "datetime.datetime.fromtimestamp",
    "datetime.datetime.utcfromtimestamp",
    "datetime.date",
    "datetime.date.today",
    "datetime.date.fromtimestamp",
    "datetime.time");

  // Method names that indicate serialization
  private static final Set<String> SERIALIZATION_METHOD_NAMES = Set.of(
    "to_dict",
    "dict",
    "asdict",
    "serialize",
    "json");

  private static final TypeMatcher LIST_TYPE = TypeMatchers.isType("builtins.list");
  private static final TypeMatcher SET_TYPE = TypeMatchers.any(TypeMatchers.isType("builtins.set"), TypeMatchers.isObjectOfType("builtins.set"));
  private static final TypeMatcher NON_SERIALIZABLE_TYPES = TypeMatchers.any(
    SET_TYPE,
    TypeMatchers.any(TypeMatchers.isType("re.Pattern"), TypeMatchers.isObjectOfType("re.Pattern")),
    TypeMatchers.any(TypeMatchers.isType("decimal.Decimal"), TypeMatchers.isObjectOfType("decimal.Decimal")),
    TypeMatchers.isObjectInstanceOf("typing.IO"),
    TypeMatchers.isObjectInstanceOf("io.IOBase"),
    TypeMatchers.any(TypeMatchers.isType("builtins.complex"), TypeMatchers.isObjectOfType("builtins.complex")),
    TypeMatchers.any(TypeMatchers.isType("builtins.bytes"), TypeMatchers.isObjectOfType("builtins.bytes")),
    TypeMatchers.any(TypeMatchers.isType("builtins.bytearray"), TypeMatchers.isObjectOfType("builtins.bytearray")),
    TypeMatchers.any(TypeMatchers.isType("builtins.frozenset"), TypeMatchers.isObjectOfType("builtins.frozenset")));

  record IssueLocation(Tree mainLocation, Optional<Tree> secondaryLocation) {
    public IssueLocation(Tree mainLocation) {
      this(mainLocation, Optional.empty());
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RETURN_STMT, this::checkReturnStatement);
  }

  private void checkReturnStatement(SubscriptionContext ctx) {
    ReturnStatement returnStmt = (ReturnStatement) ctx.syntaxNode();

    Tree parentFunction = TreeUtils.firstAncestorOfKind(returnStmt, Tree.Kind.FUNCDEF);
    if (parentFunction == null) {
      return;
    }

    FunctionDef function = (FunctionDef) parentFunction;
    if (!AwsLambdaChecksUtils.isOnlyLambdaHandler(ctx, function)) {
      return;
    }

    if (returnStmt.expressions().isEmpty()) {
      return;
    }

    Expression returnExpr = returnStmt.expressions().get(0);
    getNonSerializableExpr(returnExpr, ctx)
      .forEach(location -> {
        PreciseIssue issue = ctx.addIssue(location.mainLocation, MESSAGE);
        location.secondaryLocation.ifPresent(secondaryLocation -> issue.secondary(secondaryLocation, SECONDARY_LOCATION_MESSAGE));
      });
  }

  private Stream<IssueLocation> getNonSerializableExpr(Expression expr, SubscriptionContext ctx) {
    switch (expr.getKind()) {
      case CALL_EXPR:
        return getNonSerializableCallExpr((CallExpression) expr, ctx);
      case DICTIONARY_LITERAL:
        return getNonSerializableInDictionaryLiteral((DictionaryLiteral) expr, ctx);
      case LIST_LITERAL:
        return getNonSerializableInListLiteral((ListLiteral) expr, ctx);
      case TUPLE:
        return getNonSerializableInTuple((Tuple) expr, ctx);
      case NAME:
        return getNonSerializableFromName((Name) expr, ctx);
      default:
        return getNonSerializableFromTypeOrFqn(expr, ctx);
    }
  }

  private Stream<IssueLocation> getNonSerializableCallExpr(CallExpression callExpr, SubscriptionContext ctx) {
    // First check if this is a serialization method call like user.to_dict()
    if (isSerializationMethodCall(callExpr)) {
      return Stream.of();
    }

    Symbol calleeSymbol = callExpr.calleeSymbol();
    String fullyQualifiedName = calleeSymbol != null ? calleeSymbol.fullyQualifiedName() : null;
    if (fullyQualifiedName == null) {
      return Stream.of();
    }
    // Check if this is a list call converting non-serializable types to serializable ones
    if (LIST_TYPE.isTrueFor(callExpr.callee(), ctx)) {
      return callExpr.arguments().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .filter(argExpr -> !isSet(argExpr, ctx))
        .flatMap(expression -> getNonSerializableExpr(expression, ctx));
    }

    if (NON_SERIALIZABLE_TYPES.isTrueFor(callExpr, ctx) ||
      NON_SERIALIZABLE_FQNS.contains(fullyQualifiedName) ||
      isUserDefinedClassWithoutSerializationMethods(calleeSymbol)) {
      return Stream.of(new IssueLocation(callExpr));
    }
    return Stream.of();
  }

  private static boolean isSerializationMethodCall(CallExpression callExpr) {
    if (callExpr.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      QualifiedExpression qualifiedExpr = (QualifiedExpression) callExpr.callee();
      String methodName = qualifiedExpr.name().name();
      return SERIALIZATION_METHOD_NAMES.contains(methodName);
    }
    return false;
  }

  private static boolean isUserDefinedClassWithoutSerializationMethods(Symbol symbol) {
    return symbol.usages().stream()
      .filter(usage -> usage.kind() == Usage.Kind.CLASS_DECLARATION)
      .map(Usage::tree)
      .findFirst()
      .flatMap(TreeUtils::getSymbolFromTree)
      .filter(ClassSymbol.class::isInstance)
      .map(ClassSymbol.class::cast)
      .map(classSymbol -> !classSymbol.canHaveMember("__dict__") && !classSymbol.canHaveMember("__json__"))
      .orElse(false);
  }

  private Stream<IssueLocation> getNonSerializableInDictionaryLiteral(DictionaryLiteral dictLiteral, SubscriptionContext ctx) {
    // Check dictionary keys and values recursively
    return dictLiteral.elements().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(KeyValuePair.class))
      .flatMap(kvPair -> Stream.concat(getNonSerializableExpr(kvPair.key(), ctx), getNonSerializableExpr(kvPair.value(), ctx)));
  }

  private Stream<IssueLocation> getNonSerializableInListLiteral(ListLiteral listLiteral, SubscriptionContext ctx) {
    return listLiteral.elements().expressions().stream()
      .flatMap(expression -> getNonSerializableExpr(expression, ctx));
  }

  private Stream<IssueLocation> getNonSerializableInTuple(Tuple tuple, SubscriptionContext ctx) {
    return tuple.elements().stream()
      .flatMap(expression -> getNonSerializableExpr(expression, ctx));
  }

  private Stream<IssueLocation> getNonSerializableFromName(Name name, SubscriptionContext ctx) {
    var assignedValue = Expressions.singleAssignedValue(name);
    if (assignedValue == null) {
      return getNonSerializableFromFuncDef(name);
    }
    return getNonSerializableExpr(assignedValue, ctx)
      .map(assignedValueLocation -> new IssueLocation(name, Optional.of(assignedValueLocation.mainLocation)));
  }

  private static Stream<IssueLocation> getNonSerializableFromFuncDef(Name name) {
    return TreeUtils.getSymbolFromTree(name).stream()
      .flatMap(symbol -> symbol.usages().stream())
      .filter(usage -> usage.kind() == Usage.Kind.FUNC_DECLARATION)
      .findFirst()
      .map(usage -> new IssueLocation(name)).stream();
  }

  private static boolean isSet(Expression expr, SubscriptionContext ctx) {
    return SET_TYPE.isTrueFor(expr, ctx);
  }

  private static Stream<IssueLocation> getNonSerializableFromTypeOrFqn(Expression expr, SubscriptionContext ctx) {
    if (NON_SERIALIZABLE_TYPES.isTrueFor(expr, ctx)) {
      return Stream.of(new IssueLocation(expr));
    }

    return TreeUtils.getSymbolFromTree(expr)
      .map(Symbol::fullyQualifiedName)
      .filter(NON_SERIALIZABLE_FQNS::contains)
      .map(fqn -> new IssueLocation(expr)).stream();
  }
}
