/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.django;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6552")
public class DjangoReceiverDecoratorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Move this '@receiver' decorator to the top of the other decorators.";
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
            ctx.addIssue(receiverDecorator, MESSAGE);
          }
        });
    });
  }

  private static boolean isReceiverDecorator(Decorator decorator) {
    return Optional.of(decorator)
      .map(Decorator::expression)
      .map(TreeUtils::fullyQualifiedNameFromExpression)
      .filter(RECEIVER_DECORATOR_FQN::equals)
      .isPresent();
  }
}
