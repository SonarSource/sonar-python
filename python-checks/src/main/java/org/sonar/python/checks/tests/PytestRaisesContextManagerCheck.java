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
package org.sonar.python.checks.tests;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9000")
public class PytestRaisesContextManagerCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Wrap the code that should raise with \"with pytest.raises(ExpectedError)\".";
  private static final TypeMatcher PYTEST_PARAMETRIZE_MATCHER = TypeMatchers.withFQN("pytest.mark.parametrize");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    if (isUsedAsWithContextManager(callExpression)
      || isAssignedAndUsedInWith(callExpression)
      || isParametrizeInjectedAndUsedInWith(callExpression, ctx)
      || escapesEnclosingFunction(callExpression)) {
      return;
    }
    if (UnittestUtils.isPytestRaises(callExpression, ctx)) {
      ctx.addIssue(callExpression, MESSAGE);
    }
  }

  /**
   * Accepts {@code pytest.raises(...)} that escapes the current function for later {@code with}-usage
   * elsewhere: return, call argument, a local that is returned/passed as argument, a global/module/class
   * binding, or a {@code self} attribute assignment. Decorator/{@code parametrize} injection stays handled
   * separately so unused injected values can still be reported.
   */
  private static boolean escapesEnclosingFunction(CallExpression callExpression) {
    if (isReturned(callExpression)
      || isPassedAsCallArgument(callExpression)
      || isAssignedToSelfAttribute(callExpression)
      || isModuleOrClassLevelBinding(callExpression)) {
      return true;
    }
    return assignedName(callExpression)
      .map(Name::symbolV2)
      .filter(Objects::nonNull)
      .filter(PytestRaisesContextManagerCheck::escapesViaName)
      .isPresent();
  }

  private static boolean isReturned(CallExpression callExpression) {
    return TreeUtils.firstAncestorOfKind(outermostParenthesized(callExpression), Tree.Kind.RETURN_STMT) != null;
  }

  private static boolean isPassedAsCallArgument(CallExpression raisesCall) {
    if (TreeUtils.firstAncestorOfKind(raisesCall, Tree.Kind.DECORATOR) != null) {
      return false;
    }
    for (Tree current = outermostParenthesized(raisesCall).parent(); current != null; current = current.parent()) {
      if (current instanceof RegularArgument) {
        return true;
      }
      if (current.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA)) {
        return false;
      }
    }
    return false;
  }

  private static boolean isModuleOrClassLevelBinding(CallExpression callExpression) {
    Expression rhs = outermostParenthesized(callExpression);
    Tree parent = rhs.parent();
    if (!(parent instanceof AssignmentStatement || parent instanceof AnnotatedAssignment)) {
      return false;
    }
    return TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.FUNCDEF) == null;
  }

  private static boolean isAssignedToSelfAttribute(CallExpression callExpression) {
    Expression rhs = outermostParenthesized(callExpression);
    Tree parent = rhs.parent();
    Expression lhs = null;
    if (parent instanceof AssignmentStatement assignment) {
      if (assignment.lhsExpressions().size() != 1 || assignment.lhsExpressions().get(0).expressions().size() != 1) {
        return false;
      }
      lhs = Expressions.removeParentheses(assignment.lhsExpressions().get(0).expressions().get(0));
    } else if (parent instanceof AnnotatedAssignment annotatedAssignment
      && annotatedAssignment.assignedValue() == rhs) {
      lhs = annotatedAssignment.variable();
    }
    if (!(lhs instanceof QualifiedExpression qualifiedExpression)) {
      return false;
    }
    Expression qualifier = Expressions.removeParentheses(qualifiedExpression.qualifier());
    return qualifier instanceof Name name && "self".equals(name.name());
  }

  private static boolean escapesViaName(SymbolV2 symbol) {
    return symbol.usages().stream().anyMatch(usage ->
      usage.kind() == UsageV2.Kind.GLOBAL_DECLARATION
        || isReturnedUsage(usage)
        || isPassedAsArgumentUsage(usage));
  }

  private static boolean isReturnedUsage(UsageV2 usage) {
    return !usage.isBindingUsage()
      && TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.RETURN_STMT) != null;
  }

  private static boolean isPassedAsArgumentUsage(UsageV2 usage) {
    if (usage.isBindingUsage()) {
      return false;
    }
    for (Tree current = usage.tree().parent(); current != null; current = current.parent()) {
      if (current instanceof RegularArgument) {
        return true;
      }
      if (current.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA)) {
        return false;
      }
    }
    return false;
  }

  /**
   * Accepts direct and parenthesized context-manager expressions in {@code with} headers.
   * {@code ExitStack.enter_context(pytest.raises(...))} is not excluded yet because {@code ExitStack} is often unknown to type inference.
   */
  private static boolean isUsedAsWithContextManager(CallExpression callExpression) {
    return TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.WITH_ITEM) != null;
  }

  /**
   * Accepts {@code ctx = pytest.raises(...)} (including parenthesized RHS) when {@code ctx} is later used in a
   * {@code with} statement. Assignment that is never used as a context manager is still reported.
   */
  private static boolean isAssignedAndUsedInWith(CallExpression callExpression) {
    return assignedName(callExpression)
      .map(Name::symbolV2)
      .filter(Objects::nonNull)
      .filter(PytestRaisesContextManagerCheck::isUsedInWith)
      .isPresent();
  }

  /**
   * Accepts {@code pytest.raises(...)} nested in {@code @pytest.mark.parametrize} / {@code pytest.param} argvalues
   * when the corresponding test parameter is used in a {@code with} statement. Raises values that are never used
   * as a context manager (including a different parameter than the one used in {@code with}) are still reported.
   */
  private static boolean isParametrizeInjectedAndUsedInWith(CallExpression raisesCall, SubscriptionContext ctx) {
    CallExpression parametrizeCall = enclosingParametrizeCall(raisesCall, ctx);
    if (parametrizeCall == null) {
      return false;
    }
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(raisesCall, Tree.Kind.FUNCDEF);
    if (functionDef == null) {
      return false;
    }
    List<String> argNames = parametrizeArgNames(parametrizeCall);
    OptionalInt column = columnIndexInArgvalues(raisesCall, parametrizeCall);
    if (column.isEmpty() || column.getAsInt() >= argNames.size()) {
      return false;
    }
    return parameterSymbol(functionDef, argNames.get(column.getAsInt()))
      .filter(PytestRaisesContextManagerCheck::isUsedInWith)
      .isPresent();
  }

  private static CallExpression enclosingParametrizeCall(CallExpression raisesCall, SubscriptionContext ctx) {
    Decorator decorator = (Decorator) TreeUtils.firstAncestorOfKind(raisesCall, Tree.Kind.DECORATOR);
    if (decorator == null) {
      return null;
    }
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression parametrizeCall)) {
      return null;
    }
    return PYTEST_PARAMETRIZE_MATCHER.isTrueFor(parametrizeCall.callee(), ctx) ? parametrizeCall : null;
  }

  private static List<String> parametrizeArgNames(CallExpression parametrizeCall) {
    RegularArgument argNamesArgument = TreeUtils.nthArgumentOrKeyword(0, "argnames", parametrizeCall.arguments());
    if (argNamesArgument == null) {
      return List.of();
    }
    Expression expression = Expressions.removeParentheses(argNamesArgument.expression());
    if (expression instanceof StringLiteral stringLiteral) {
      return Arrays.stream(stringLiteral.trimmedQuotesValue().split(","))
        .map(String::trim)
        .filter(name -> !name.isEmpty())
        .toList();
    }
    return Expressions.expressionsFromListOrTuple(expression).stream()
      .map(Expressions::removeParentheses)
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .map(String::trim)
      .filter(name -> !name.isEmpty())
      .toList();
  }

  private static OptionalInt columnIndexInArgvalues(CallExpression raisesCall, CallExpression parametrizeCall) {
    RegularArgument argValuesArgument = TreeUtils.nthArgumentOrKeyword(1, "argvalues", parametrizeCall.arguments());
    if (argValuesArgument == null) {
      return OptionalInt.empty();
    }
    Expression argValues = Expressions.removeParentheses(argValuesArgument.expression());
    for (Expression row : Expressions.expressionsFromListOrTuple(argValues)) {
      OptionalInt column = columnIndexInRow(raisesCall, Expressions.removeParentheses(row));
      if (column.isPresent()) {
        return column;
      }
    }
    return OptionalInt.empty();
  }

  private static OptionalInt columnIndexInRow(CallExpression raisesCall, Expression row) {
    if (row == raisesCall) {
      return OptionalInt.of(0);
    }
    List<Expression> elements = Expressions.expressionsFromListOrTuple(row);
    if (!elements.isEmpty()) {
      return indexOfContainingExpression(raisesCall, elements);
    }
    return columnIndexInCallPositionalArguments(raisesCall, row);
  }

  private static OptionalInt columnIndexInCallPositionalArguments(CallExpression raisesCall, Expression row) {
    if (!(row instanceof CallExpression call)) {
      return OptionalInt.empty();
    }
    List<Expression> positionalArguments = call.arguments().stream()
      .map(PytestRaisesContextManagerCheck::positionalArgumentExpression)
      .flatMap(Optional::stream)
      .toList();
    return indexOfContainingExpression(raisesCall, positionalArguments);
  }

  private static Optional<Expression> positionalArgumentExpression(Argument argument) {
    if (!(argument instanceof RegularArgument regularArgument) || regularArgument.keywordArgument() != null) {
      return Optional.empty();
    }
    return Optional.of(regularArgument.expression());
  }

  private static OptionalInt indexOfContainingExpression(CallExpression raisesCall, List<Expression> expressions) {
    for (int i = 0; i < expressions.size(); i++) {
      if (expressionContains(expressions.get(i), raisesCall)) {
        return OptionalInt.of(i);
      }
    }
    return OptionalInt.empty();
  }

  private static boolean expressionContains(Expression expression, CallExpression raisesCall) {
    Expression unwrapped = Expressions.removeParentheses(expression);
    return unwrapped == raisesCall || TreeUtils.hasDescendant(unwrapped, tree -> tree == raisesCall);
  }

  private static Optional<SymbolV2> parameterSymbol(FunctionDef functionDef, String paramName) {
    ParameterList parameters = functionDef.parameters();
    if (parameters == null) {
      return Optional.empty();
    }
    return parameters.nonTuple().stream()
      .map(Parameter::name)
      .filter(Objects::nonNull)
      .filter(name -> paramName.equals(name.name()))
      .map(Name::symbolV2)
      .filter(Objects::nonNull)
      .findFirst();
  }

  private static Optional<Name> assignedName(CallExpression callExpression) {
    Expression rhs = outermostParenthesized(callExpression);
    Tree parent = rhs.parent();
    if (parent instanceof AssignmentStatement assignment) {
      return singleAssignedName(assignment.lhsExpressions());
    }
    if (parent instanceof AnnotatedAssignment annotatedAssignment
      && annotatedAssignment.assignedValue() == rhs) {
      Expression variable = annotatedAssignment.variable();
      return variable instanceof Name name ? Optional.of(name) : Optional.empty();
    }
    return Optional.empty();
  }

  private static Expression outermostParenthesized(Expression expression) {
    Expression current = expression;
    while (current.parent() != null && current.parent().is(Tree.Kind.PARENTHESIZED)) {
      current = (Expression) current.parent();
    }
    return current;
  }

  private static Optional<Name> singleAssignedName(List<ExpressionList> lhsExpressions) {
    if (lhsExpressions.size() != 1 || lhsExpressions.get(0).expressions().size() != 1) {
      return Optional.empty();
    }
    Expression lhs = Expressions.removeParentheses(lhsExpressions.get(0).expressions().get(0));
    return lhs instanceof Name name ? Optional.of(name) : Optional.empty();
  }

  private static boolean isUsedInWith(SymbolV2 symbol) {
    return symbol.usages().stream()
      .filter(usage -> !usage.isBindingUsage())
      .map(UsageV2::tree)
      .anyMatch(tree -> TreeUtils.firstAncestorOfKind(tree, Tree.Kind.WITH_ITEM) != null);
  }
}
