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

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NUMERIC_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.QUALIFIED_EXPR;
import static org.sonar.python.checks.utils.CheckUtils.isConstant;

@Rule(key = "S5914")
public class UnconditionalAssertionCheck extends PythonSubscriptionCheck {

  private static final Set<String> BOOLEAN_ASSERTIONS = Set.of("assertTrue", "assertFalse");
  private static final Set<String> NONE_ASSERTIONS = Set.of("assertIsNone", "assertIsNotNone");
  private static final Set<String> IS_ASSERTIONS = Set.of("assertIs", "assertIsNot");

  private static final String BOOLEAN_MESSAGE = "Replace this expression; its boolean value is constant.";
  private static final String NONE_MESSAGE = "Remove this identity assertion; its value is constant.";
  private static final String IS_MESSAGE = "Replace this \"assertIs\" call with an \"assertEqual\" call.";
  private static final String IS_NOT_MESSAGE = "Replace this \"assertIsNot\" call with an \"assertNotEqual\" call.";
  private static final String IS_SECONDARY_MESSAGE = "This expression creates a new object every time.";

  private static final Set<String> ACCEPTED_DECORATORS = Set.of("overload", "staticmethod", "classmethod");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> {
      AssertStatement assertStatement = (AssertStatement) ctx.syntaxNode();
      Expression condition = assertStatement.condition();
      if (!condition.is(Tree.Kind.TUPLE) && !isFalseOrZeroLiteral(condition) && CheckUtils.isConstant(condition)) {
        ctx.addIssue(condition, BOOLEAN_MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression call = (CallExpression) ctx.syntaxNode();
      if (!isAssertionCall(call.callee())) {
        return;
      }
      String name = calleeName(call.callee());
      if (name == null) {
        return;
      }

      List<Argument> arguments = call.arguments();

      if (BOOLEAN_ASSERTIONS.contains(name)) {
        checkAssertion(arg -> checkBooleanAssertion(ctx, arg), 0, "expr", arguments);
      } else if (NONE_ASSERTIONS.contains(name)) {
        checkAssertion(arg -> checkNoneAssertion(ctx, call, arg),0, "expr", arguments);
      } else if (IS_ASSERTIONS.contains(name)) {
        String message = "assertIs".equals(name) ? IS_MESSAGE : IS_NOT_MESSAGE;
        checkAssertion(arg -> checkIsAssertion(ctx, call, arg, message), 0, "first", arguments);
        checkAssertion(arg -> checkIsAssertion(ctx, call, arg, message),1, "second", arguments);
      }
    });
  }

  private static @Nullable String calleeName(Expression callee) {
    return switch (callee) {
      case Name name -> name.name();
      case QualifiedExpression qualifiedExpression -> qualifiedExpression.name().name();
      default -> null;
    };
  }

  private static boolean isAssertionCall(Expression callee) {
    if (!isSupportedAssertion(calleeName(callee))) {
      return false;
    }
    if (callee.typeV2() instanceof FunctionType) {
      return true;
    }
    return callee instanceof QualifiedExpression qualifiedExpression
      && CheckUtils.isSelf(qualifiedExpression.qualifier())
      && isInTestCase(qualifiedExpression);
  }

  private static boolean isSupportedAssertion(@Nullable String name) {
    return name != null && (BOOLEAN_ASSERTIONS.contains(name) || NONE_ASSERTIONS.contains(name) || IS_ASSERTIONS.contains(name));
  }

  private static boolean isInTestCase(QualifiedExpression qualifiedExpression) {
    ClassDef classDef = (ClassDef) TreeUtils.firstAncestorOfKind(qualifiedExpression, Tree.Kind.CLASSDEF);
    return classDef != null && (isTestClass(classDef) || hasTestCaseBase(classDef, new HashSet<>()));
  }

  private static boolean isTestClass(ClassDef classDef) {
    String className = classDef.name().name();
    return className.endsWith("Test") || className.endsWith("Tests");
  }

  private static boolean hasTestCaseBase(ClassDef classDef, Set<ClassDef> visitedClasses) {
    if (!visitedClasses.add(classDef) || classDef.args() == null) {
      return false;
    }
    return classDef.args().arguments().stream()
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .filter(argument -> argument.keywordArgument() == null)
      .map(RegularArgument::expression)
      .anyMatch(baseClass -> isTestCaseBase(baseClass)
        || findClassDef(baseClass).map(parentClass -> hasTestCaseBase(parentClass, visitedClasses)).orElse(false));
  }

  private static boolean isTestCaseBase(Expression expression) {
    return switch (expression) {
      case Name name -> name.name().endsWith("TestCase");
      case QualifiedExpression qualifiedExpression -> qualifiedExpression.name().name().endsWith("TestCase");
      default -> false;
    };
  }

  private static Optional<ClassDef> findClassDef(Expression expression) {
    if (!(expression instanceof Name name) || name.symbolV2() == null) {
      return Optional.empty();
    }
    return name.symbolV2().usages().stream()
      .filter(usage -> usage.kind() == UsageV2.Kind.CLASS_DECLARATION)
      .map(UsageV2::tree)
      .map(tree -> TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CLASSDEF))
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .findFirst();
  }

  private static void checkAssertion(Consumer<RegularArgument> checkConsumer, int argPosition, String keyword, List<Argument> arguments) {
    Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(argPosition, keyword, arguments)).ifPresent(checkConsumer);
  }

  /**
   * `assert False` or `assert 0` is often used to make a test fail.
   * Usually it is better to use another assertion or throw an AssertionException.
   * However, this rule is not intended to check this best practice.
   */
  private static boolean isFalseOrZeroLiteral(Expression expression) {
    if (expression.is(NAME)) {
      return "False".equals(((Name) expression).name());
    }
    if (expression.is(NUMERIC_LITERAL)) {
      try {
        return ((NumericLiteral) expression).valueAsLong() == 0;
      } catch (NumberFormatException nfe) {
        return false;
      }
    }
    return false;
  }

  private static void checkNoneAssertion(SubscriptionContext ctx, CallExpression call, RegularArgument arg) {
    if (isUnconditional(arg, ctx)) {
      ctx.addIssue(call, NONE_MESSAGE);
    }
  }

  private static void checkBooleanAssertion(SubscriptionContext ctx, RegularArgument arg) {
    if (isUnconditional(arg, ctx)) {
      ctx.addIssue(arg, BOOLEAN_MESSAGE);
    }
  }

  private static void checkIsAssertion(SubscriptionContext ctx, CallExpression call, RegularArgument arg, String message) {
    if (CheckUtils.isConstantCollectionLiteral(arg.expression())) {
      ctx.addIssue(call.callee(), message).secondary(arg, IS_SECONDARY_MESSAGE);
    }
  }

  private static boolean isUnconditional(RegularArgument argument, SubscriptionContext ctx) {
    Expression expression = argument.expression();
    if (isConstant(expression)) {
      return true;
    }

    if (expression.is(NAME) || expression.is(QUALIFIED_EXPR)) {
      if (isClassOrFunction(expression)) {
        return true;
      }
      if (expression instanceof Name name && hasAnyNonlocalStatement(name)) {
        return false;
      }
    }

    if (expression.is(NAME)) {
      Set<Expression> valuesAtLocation = ctx.valuesAtLocation(((Name) expression));
      if (valuesAtLocation.size() == 1) {
        return CheckUtils.isImmutableConstant(valuesAtLocation.iterator().next());
      }
    }

    return false;
  }

  private static boolean hasAnyNonlocalStatement(Name name) {
    return name.symbolV2() != null && name.symbolV2().usages().stream()
      .anyMatch(usage -> usage.tree().parent().is(Tree.Kind.NONLOCAL_STMT));
  }

  private static boolean isClassOrFunction(Expression expression) {
    PythonType type = expression.typeV2();
    if (type instanceof ClassType) {
      return true;
    }
    if (type instanceof FunctionType functionType) {
      // Avoid potential FPs with properties: only report on limited selection of "safe" decorators
      return functionType.decorators().stream()
        .map(TypeWrapper::type)
        .map(PythonType::name)
        .allMatch(ACCEPTED_DECORATORS::contains);
    }
    return false;
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
