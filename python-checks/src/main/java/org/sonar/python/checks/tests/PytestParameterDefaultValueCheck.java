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
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9106")
public class PytestParameterDefaultValueCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this default value so pytest can inject the parameter.";
  private static final String QUICK_FIX_MESSAGE = "Remove default value";
  private static final String FIXTURE_NAME_ARGUMENT = "name";
  private static final String PARAMETRIZE_ARGNAMES = "argnames";
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN(UnittestUtils.PYTEST_FIXTURE_DECORATOR_FQN);
  private static final TypeMatcher PYTEST_PARAMETRIZE_MATCHER = TypeMatchers.isType("pytest.mark.parametrize");

  /**
   * Core pytest builtin fixture names. A default on a parameter with one of these names always
   * blocks injection of that builtin fixture.
   */
  private static final Set<String> BUILTIN_FIXTURES = Set.of(
    "request",
    "tmp_path",
    "tmpdir",
    "tmp_path_factory",
    "tmpdir_factory",
    "monkeypatch",
    "capsys",
    "capfd",
    "capfdbinary",
    "capsysbinary",
    "caplog",
    "cache",
    "doctest_namespace",
    "pytestconfig",
    "record_property",
    "record_testsuite_property",
    "recwarn",
    "testdir",
    "pytester");

  private final Set<String> moduleFixtures = new HashSet<>();
  private final Map<ClassDef, Set<String>> classFixtures = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::collectFixtures);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkFunctionDef);
  }

  private void collectFixtures(SubscriptionContext ctx) {
    moduleFixtures.clear();
    classFixtures.clear();
    moduleFixtures.addAll(BUILTIN_FIXTURES);

    FileInput fileInput = (FileInput) ctx.syntaxNode();
    if (fileInput.statements() == null) {
      return;
    }
    for (Statement statement : fileInput.statements().statements()) {
      collectFixtureFromStatement(statement, ctx);
    }
  }

  private void collectFixtureFromStatement(Statement statement, SubscriptionContext ctx) {
    if (statement instanceof FunctionDef functionDef) {
      addModuleFixture(functionDef, ctx);
    } else if (statement instanceof ClassDef classDef) {
      collectClassFixtures(classDef, ctx);
    }
  }

  private void addModuleFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    String name = fixtureName(functionDef, ctx);
    if (name != null) {
      moduleFixtures.add(name);
    }
  }

  private void collectClassFixtures(ClassDef classDef, SubscriptionContext ctx) {
    Set<String> fixtures = new HashSet<>();
    for (Statement bodyStatement : classDef.body().statements()) {
      if (bodyStatement instanceof FunctionDef methodDef) {
        String name = fixtureName(methodDef, ctx);
        if (name != null) {
          fixtures.add(name);
        }
      }
    }
    if (!fixtures.isEmpty()) {
      classFixtures.put(classDef, fixtures);
    }
  }

  private void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!isPytestTestOrFixture(functionDef, ctx)) {
      return;
    }

    ParameterList parameterList = functionDef.parameters();
    if (parameterList == null) {
      return;
    }

    Set<String> injectedNames = injectedParameterNames(functionDef, ctx);
    for (Parameter parameter : parameterList.nonTuple()) {
      reportDefaultIfInjected(ctx, parameter, injectedNames);
    }
  }

  /**
   * Names pytest will inject for this function: known fixtures in scope, plus statically known
   * {@code @pytest.mark.parametrize} argnames.
   */
  private Set<String> injectedParameterNames(FunctionDef functionDef, SubscriptionContext ctx) {
    Set<String> injected = new HashSet<>(moduleFixtures);
    ClassDef enclosingClass = CheckUtils.getParentClassDef(functionDef);
    if (enclosingClass != null) {
      Set<String> fixtures = classFixtures.get(enclosingClass);
      if (fixtures != null) {
        injected.addAll(fixtures);
      }
    }
    injected.addAll(parametrizeArgNames(functionDef, ctx));
    return injected;
  }

  private static Set<String> parametrizeArgNames(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .flatMap(decorator -> parametrizeArgNames(decorator, ctx))
      .collect(Collectors.toSet());
  }

  private static Stream<String> parametrizeArgNames(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression callExpression)
      || !PYTEST_PARAMETRIZE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return Stream.empty();
    }
    RegularArgument argNamesArgument = TreeUtils.nthArgumentOrKeyword(0, PARAMETRIZE_ARGNAMES, callExpression.arguments());
    if (argNamesArgument == null) {
      return Stream.empty();
    }
    Expression argNames = Expressions.removeParentheses(argNamesArgument.expression());
    if (argNames instanceof StringLiteral stringLiteral) {
      return Arrays.stream(stringLiteral.trimmedQuotesValue().split(","))
        .map(String::trim)
        .filter(name -> !name.isEmpty());
    }
    return Expressions.expressionsFromListOrTuple(argNames).stream()
      .map(Expressions::removeParentheses)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue)
      .map(String::trim)
      .filter(name -> !name.isEmpty());
  }

  private static void reportDefaultIfInjected(SubscriptionContext ctx, Parameter parameter, Set<String> injectedNames) {
    if (parameter.starToken() != null || parameter.name() == null) {
      return;
    }
    Expression defaultValue = parameter.defaultValue();
    if (defaultValue == null) {
      return;
    }
    String parameterName = parameter.name().name();
    if (!injectedNames.contains(parameterName)) {
      return;
    }

    var issue = ctx.addIssue(defaultValue, MESSAGE);
    Token previousToken = parameter.typeAnnotation() != null
      ? parameter.typeAnnotation().lastToken()
      : parameter.name().lastToken();
    issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.removeRange(
        previousToken.pythonLine(),
        endColumn(previousToken),
        defaultValue.lastToken().pythonLine(),
        endColumn(defaultValue.lastToken())))
      .build());
  }

  private static boolean isPytestTestOrFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return isPytestFixture(functionDef, ctx)
      || UnittestUtils.isCollectedPytestTestFunction(functionDef, ctx.pythonFile().fileName());
  }

  private static boolean isPytestFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream()
      .anyMatch(decorator -> PYTEST_FIXTURE_MATCHER.isTrueFor(decoratorFunctionExpression(decorator), ctx));
  }

  /**
   * Returns the pytest fixture name for this function, honoring {@code @pytest.fixture(name=...)},
   * or {@code null} if the function is not a fixture.
   */
  private static String fixtureName(FunctionDef functionDef, SubscriptionContext ctx) {
    for (Decorator decorator : functionDef.decorators()) {
      Expression expression = decorator.expression();
      Expression callee = decoratorFunctionExpression(decorator);
      if (!PYTEST_FIXTURE_MATCHER.isTrueFor(callee, ctx)) {
        continue;
      }
      if (expression instanceof CallExpression callExpression) {
        RegularArgument nameArgument = TreeUtils.argumentByKeyword(FIXTURE_NAME_ARGUMENT, callExpression.arguments());
        if (nameArgument != null && nameArgument.expression() instanceof StringLiteral stringLiteral) {
          return stringLiteral.trimmedQuotesValue();
        }
      }
      return functionDef.name().name();
    }
    return null;
  }

  private static Expression decoratorFunctionExpression(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return callExpression.callee();
    }
    return expression;
  }

  private static int endColumn(Token token) {
    return token.pythonColumn() + token.value().length();
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
