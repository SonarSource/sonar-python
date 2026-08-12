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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.MockPatchUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9137")
public class MocksShouldUseAutospecCheck extends PythonSubscriptionCheck {

  private static final String MOCK_MESSAGE_FORMAT =
    "Replace this \"%s()\" with \"create_autospec(<collaborator>)\", or pass \"spec=\" / \"spec_set=\".";
  private static final String PATCH_MESSAGE =
    "Add \"autospec=True\" to this patch call, or pass an explicit \"spec=\" / \"spec_set=\".";
  private static final String AUTOSPEC_FALSE_MESSAGE =
    "Replace \"autospec=False\" with \"autospec=True\", or pass an explicit \"spec=\" / \"spec_set=\".";

  private static final String SPEC = "spec";
  private static final String SPEC_SET = "spec_set";
  private static final String AUTOSPEC = "autospec";
  private static final String NEW = "new";
  private static final String NEW_CALLABLE = "new_callable";
  private static final String SELF = "self";
  private static final String CLS = "cls";

  /**
   * unittest.mock helpers / assertions — invoking these does not mean the test exercises a collaborator API
   * that would benefit from autospeccing.
   */
  private static final Set<String> MOCK_HELPER_METHODS = Set.of(
    "assert_called",
    "assert_called_once",
    "assert_called_with",
    "assert_called_once_with",
    "assert_any_call",
    "assert_has_calls",
    "assert_not_called",
    "assert_never_called",
    "reset_mock",
    "attach_mock",
    "configure_mock");

  private static final TypeMatcher MOCK_CLASS_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("unittest.mock.Mock"),
    TypeMatchers.isType("unittest.mock.MagicMock"),
    TypeMatchers.isType("mock.mock.Mock"),
    TypeMatchers.isType("mock.mock.MagicMock"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, MocksShouldUseAutospecCheck::checkCall);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    List<Argument> arguments = callExpression.arguments();
    if (Expressions.containsSpreadOperator(arguments)) {
      return;
    }

    Expression callee = Expressions.removeParentheses(callExpression.callee());
    if (MOCK_CLASS_MATCHER.isTrueFor(callee, ctx)) {
      if (!hasMockSpec(arguments) && mockBenefitsFromAutospec(callExpression, ctx)) {
        ctx.addIssue(callee, mockMessage(callee));
      }
      return;
    }

    int newPosition = MockPatchUtils.newArgumentPosition(callee, ctx);
    if (newPosition < 0) {
      return;
    }
    if (hasReplacement(arguments, newPosition) || hasPatchSpecOrAutospec(arguments, newPosition)) {
      return;
    }
    if (!patchMockIsUsedInTest(callExpression, ctx)) {
      return;
    }
    ctx.addIssue(callee, patchMessage(arguments, newPosition));
  }

  private static String mockMessage(Expression callee) {
    return String.format(MOCK_MESSAGE_FORMAT, mockConstructorName(callee));
  }

  private static String mockConstructorName(Expression callee) {
    if (callee instanceof Name name) {
      return name.name();
    }
    if (callee instanceof QualifiedExpression qualifiedExpression) {
      return qualifiedExpression.name().name();
    }
    return "Mock";
  }

  private static String patchMessage(List<Argument> arguments, int newPosition) {
    int autospecPosition = newPosition + 4;
    RegularArgument autospecArgument = TreeUtils.nthArgumentOrKeyword(autospecPosition, AUTOSPEC, arguments);
    if (autospecArgument != null && Expressions.isFalsy(autospecArgument.expression())) {
      return AUTOSPEC_FALSE_MESSAGE;
    }
    return PATCH_MESSAGE;
  }

  private static boolean hasReplacement(List<Argument> arguments, int newPosition) {
    return TreeUtils.nthArgumentOrKeyword(newPosition, NEW, arguments) != null
      || TreeUtils.argumentByKeyword(NEW_CALLABLE, arguments) != null;
  }

  private static boolean hasMockSpec(List<Argument> arguments) {
    // Mock(spec, wraps, name, spec_set, ...)
    return TreeUtils.nthArgumentOrKeyword(0, SPEC, arguments) != null
      || TreeUtils.nthArgumentOrKeyword(3, SPEC_SET, arguments) != null;
  }

  private static boolean hasPatchSpecOrAutospec(List<Argument> arguments, int newPosition) {
    // patch(target, new, spec, create, spec_set, autospec, new_callable)
    // patch.object(target, attribute, new, spec, create, spec_set, autospec, new_callable)
    int specPosition = newPosition + 1;
    int specSetPosition = newPosition + 3;
    int autospecPosition = newPosition + 4;
    return TreeUtils.nthArgumentOrKeyword(specPosition, SPEC, arguments) != null
      || TreeUtils.nthArgumentOrKeyword(specSetPosition, SPEC_SET, arguments) != null
      || hasTruthyAutospec(arguments, autospecPosition);
  }

  private static boolean hasTruthyAutospec(List<Argument> arguments, int autospecPosition) {
    RegularArgument autospecArgument = TreeUtils.nthArgumentOrKeyword(autospecPosition, AUTOSPEC, arguments);
    return autospecArgument != null && !Expressions.isFalsy(autospecArgument.expression());
  }

  /**
   * Raise only when the mock is used as a collaborator in the test (passed into a non-mock call, or a
   * non-helper method is invoked on it). Field-only stubs cannot be autospecced reliably.
   */
  private static boolean mockBenefitsFromAutospec(CallExpression mockConstruction, SubscriptionContext ctx) {
    List<Name> boundNames = simpleNameBindings(mockConstruction);
    if (boundNames.isEmpty()) {
      // Attribute assignment (`obj.field = Mock()`) or a discarded call: no collaborator API to mirror.
      return isArgumentToNonMockCall(mockConstruction, ctx);
    }
    for (Name boundName : boundNames) {
      SymbolV2 symbol = boundName.symbolV2();
      if (symbol != null && symbol.usages().stream()
        .filter(usage -> !usage.isBindingUsage())
        .anyMatch(usage -> isCollaboratorUsage(usage.tree(), ctx))) {
        return true;
      }
    }
    return false;
  }

  /**
   * Simple name bindings only. Attribute / subscription LHS is field stubbing.
   * Supports plain, chained (`a = m = Mock()`), and annotated (`name: T = Mock()`) assignments.
   */
  private static List<Name> simpleNameBindings(CallExpression mockConstruction) {
    AssignmentStatement assignment = (AssignmentStatement) TreeUtils.firstAncestorOfKind(mockConstruction, Tree.Kind.ASSIGNMENT_STMT);
    if (assignment != null && Expressions.removeParentheses(assignment.assignedValue()) == mockConstruction) {
      List<Name> names = new ArrayList<>();
      for (ExpressionList target : assignment.lhsExpressions()) {
        List<Expression> exprs = target.expressions();
        // Skip tuple unpacking (`a, b = Mock()`): only single-name targets are true bindings.
        if (exprs.size() != 1 || !(exprs.get(0) instanceof Name name)) {
          return List.of();
        }
        names.add(name);
      }
      return names;
    }
    AnnotatedAssignment annotated = (AnnotatedAssignment) TreeUtils.firstAncestorOfKind(mockConstruction, Tree.Kind.ANNOTATED_ASSIGNMENT);
    if (annotated != null) {
      Expression assignedValue = annotated.assignedValue();
      if (assignedValue != null
        && Expressions.removeParentheses(assignedValue) == mockConstruction
        && annotated.variable() instanceof Name name) {
        return List.of(name);
      }
    }
    return List.of();
  }

  private static boolean isCollaboratorUsage(Tree usageTree, SubscriptionContext ctx) {
    Name name = (Name) usageTree;
    return isArgumentToNonMockCall(rootQualifierOrSelf(name), ctx) || isNonHelperMethodCallOnMock(name);
  }

  private static Expression rootQualifierOrSelf(Name name) {
    Tree cursor = name;
    while (cursor.parent() instanceof QualifiedExpression qualifiedExpression
      && qualifiedExpression.qualifier() == cursor) {
      cursor = qualifiedExpression;
    }
    return (Expression) cursor;
  }

  private static boolean isArgumentToNonMockCall(Expression expression, SubscriptionContext ctx) {
    if (!(expression.parent() instanceof RegularArgument)) {
      return false;
    }
    CallExpression enclosingCall = (CallExpression) TreeUtils.firstAncestorOfKind(expression.parent(), Tree.Kind.CALL_EXPR);
    return !MOCK_CLASS_MATCHER.isTrueFor(Expressions.removeParentheses(enclosingCall.callee()), ctx);
  }

  private static boolean isNonHelperMethodCallOnMock(Name name) {
    if (name.parent() instanceof CallExpression directCall && directCall.callee() == name) {
      return true;
    }
    Tree cursor = name;
    while (cursor.parent() instanceof QualifiedExpression qualifiedExpression
      && qualifiedExpression.qualifier() == cursor) {
      cursor = qualifiedExpression;
    }
    if (!(cursor instanceof QualifiedExpression qualifiedExpression)) {
      return false;
    }
    if (!(cursor.parent() instanceof CallExpression callExpression) || callExpression.callee() != cursor) {
      return false;
    }
    return !MOCK_HELPER_METHODS.contains(qualifiedExpression.name().name());
  }

  /**
   * Isolation-only patches (unused mock) do not benefit from autospec.
   */
  private static boolean patchMockIsUsedInTest(CallExpression patchCall, SubscriptionContext ctx) {
    Decorator decorator = (Decorator) TreeUtils.firstAncestorOfKind(patchCall, Tree.Kind.DECORATOR);
    if (decorator != null) {
      if (decorator.parent() instanceof ClassDef) {
        // Class-level @patch injects mocks into every method — keep raising.
        return true;
      }
      FunctionDef functionDef = Objects.requireNonNull(
        (FunctionDef) TreeUtils.firstAncestorOfKind(decorator, Tree.Kind.FUNCDEF));
      SymbolV2 mockSymbol = mockSymbolFromDecorator(functionDef, decorator, ctx);
      return mockSymbol != null && hasNonBindingUsage(mockSymbol);
    }

    WithItem withItem = (WithItem) TreeUtils.firstAncestorOfKind(patchCall, Tree.Kind.WITH_ITEM);
    if (withItem != null && isPatchWithItem(withItem, patchCall)) {
      if (!(withItem.expression() instanceof Name name)) {
        return false;
      }
      SymbolV2 symbol = name.symbolV2();
      return symbol != null && hasNonBindingUsage(symbol);
    }

    return Expressions.getAssignedName(patchCall)
      .map(Name::symbolV2)
      .filter(MocksShouldUseAutospecCheck::hasNonBindingUsage)
      .isPresent();
  }

  private static boolean isPatchWithItem(WithItem withItem, CallExpression patchCall) {
    return Expressions.removeParentheses(withItem.test()) == patchCall;
  }

  private static boolean hasNonBindingUsage(SymbolV2 symbol) {
    return symbol.usages().stream().anyMatch(usage -> !usage.isBindingUsage());
  }

  @CheckForNull
  private static SymbolV2 mockSymbolFromDecorator(FunctionDef functionDef, Decorator decorator, SubscriptionContext ctx) {
    List<Decorator> patchDecorators = new ArrayList<>();
    for (Decorator candidate : functionDef.decorators()) {
      if (isPatchDecorator(candidate, ctx)) {
        patchDecorators.add(candidate);
      }
    }
    int indexFromTop = patchDecorators.indexOf(decorator);
    // Bottom decorator maps to the first injected mock parameter.
    int indexFromBottom = patchDecorators.size() - 1 - indexFromTop;
    List<Parameter> mockParameters = mockParameters(functionDef);
    if (indexFromBottom >= mockParameters.size()) {
      return null;
    }
    Name parameterName = mockParameters.get(indexFromBottom).name();
    return parameterName != null ? parameterName.symbolV2() : null;
  }

  private static boolean isPatchDecorator(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = Expressions.removeParentheses(decorator.expression());
    return expression instanceof CallExpression callExpression
      && MockPatchUtils.newArgumentPosition(callExpression.callee(), ctx) >= 0;
  }

  private static List<Parameter> mockParameters(FunctionDef functionDef) {
    if (functionDef.parameters() == null) {
      return List.of();
    }
    List<Parameter> result = new ArrayList<>();
    for (Parameter parameter : functionDef.parameters().nonTuple()) {
      Name parameterName = parameter.name();
      if (parameterName != null && !SELF.equals(parameterName.name()) && !CLS.equals(parameterName.name())) {
        result.add(parameter);
      }
    }
    return result;
  }
}
