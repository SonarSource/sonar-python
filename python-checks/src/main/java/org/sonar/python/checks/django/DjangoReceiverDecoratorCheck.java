/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks.django;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6552")
public class DjangoReceiverDecoratorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Move this '@receiver' decorator to the top of the other decorators.";
  private static final String QUICK_FIX_MESSAGE = "Move the '@receiver' decorator to the top";
  private static final String RECEIVER_DECORATOR_FQN = "django.dispatch.receiver";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      var functionDef = (FunctionDef) ctx.syntaxNode();
      var decorators = functionDef.decorators();
      decorators
        .stream()
        .filter(DjangoReceiverDecoratorCheck::isReceiverDecorator)
        .findFirst()
        .ifPresent(receiverDecorator -> {
          if (decorators.indexOf(receiverDecorator) != 0) {
            var issue = ctx.addIssue(receiverDecorator, MESSAGE);
            addQuickFix(issue, functionDef, receiverDecorator);
          }
        });
    });
  }

  private static boolean isReceiverDecorator(Decorator decorator) {
    return Optional.of(decorator)
      .map(Decorator::expression)
      .flatMap(TreeUtils::fullyQualifiedNameFromExpression)
      .filter(RECEIVER_DECORATOR_FQN::equals)
      .isPresent();
  }

  private static void addQuickFix(PreciseIssue issue, FunctionDef functionDef, Decorator decorator) {
    var builder = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE);

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
