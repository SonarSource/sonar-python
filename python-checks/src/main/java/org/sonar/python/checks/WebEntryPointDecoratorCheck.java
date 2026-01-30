/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6552")
public class WebEntryPointDecoratorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Move this '%s' decorator to the top of the other decorators.";
  private static final String QUICK_FIX_TEMPLATE = "Move the '%s' decorator to the top";

  private record DecoratorMatcher(TypeMatcher matcher, String decoratorName) {}

  private static final List<DecoratorMatcher> DECORATOR_MATCHERS = List.of(
    new DecoratorMatcher(TypeMatchers.withFQN("django.dispatch.receiver"), "@receiver"),
    new DecoratorMatcher(TypeMatchers.any(
      TypeMatchers.isType("flask.app.Flask.route"),
      TypeMatchers.isType("flask.blueprints.Blueprint.route")
    ), "@route")
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, WebEntryPointDecoratorCheck::checkFunctionDef);
  }

  private static void checkFunctionDef(SubscriptionContext ctx) {
    var functionDef = (FunctionDef) ctx.syntaxNode();
    var decorators = functionDef.decorators();

    if (decorators.size() < 2) {
      return;
    }

    for (var matcher : DECORATOR_MATCHERS) {
      decorators.stream()
        .filter(decorator -> matchesDecorator(decorator, matcher.matcher(), ctx))
        .findFirst()
        .ifPresent(decorator -> checkDecoratorPosition(ctx, functionDef, decorator, matcher.decoratorName()));
    }
  }

  private static boolean matchesDecorator(Decorator decorator, TypeMatcher matcher, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (!(expression instanceof CallExpression callExpr)) {
      return false;
    }
    return matcher.isTrueFor(callExpr.callee(), ctx);
  }

  private static void checkDecoratorPosition(SubscriptionContext ctx, FunctionDef functionDef, Decorator decorator, String decoratorName) {
    var decorators = functionDef.decorators();
    if (decorators.indexOf(decorator) != 0) {
      String message = String.format(MESSAGE_TEMPLATE, decoratorName);
      String quickFixMessage = String.format(QUICK_FIX_TEMPLATE, decoratorName);
      var issue = ctx.addIssue(decorator, message);
      addQuickFix(issue, functionDef, decorator, quickFixMessage);
    }
  }

  private static void addQuickFix(PreciseIssue issue, FunctionDef functionDef, Decorator decorator, String quickFixMessage) {
    var builder = PythonQuickFix.newQuickFix(quickFixMessage);

    var decorators = functionDef.decorators();
    var decoratorPosition = decorators.indexOf(decorator);
    var removeTo = decorators.size() == decoratorPosition + 1 ? functionDef.defKeyword()
      : decorators.get(decoratorPosition + 1);

    Optional.of(decorator)
      .map(d -> TreeUtils.treeToString(d, true))
      .map(decoratorString -> TextEditUtils.insertBefore(decorators.get(0), decoratorString))
      .ifPresent(builder::addTextEdit);

    var removeEdit = TextEditUtils.removeUntil(decorator, removeTo);
    builder.addTextEdit(removeEdit);

    issue.addQuickFix(builder.build());
  }
}

