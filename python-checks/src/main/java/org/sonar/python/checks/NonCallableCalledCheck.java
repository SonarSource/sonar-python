/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S5756")
public class NonCallableCalledCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Expression callee = callExpression.callee();
      if (!callee.type().canHaveMember("__call__")) {
        Optional<String> nameOptional = nameFromExpression(callee);
        if (nameOptional.isPresent()) {
          ctx.addIssue(callee, String.format("Fix this call; \"%s\" is not callable.", nameOptional.get()));
        } else {
          ctx.addIssue(callee, "Fix this call; this expression is not callable.");
        }
      }
    });
  }

  private static Optional<String> nameFromExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Optional.of(((Name) expression).name());
    }
    return Optional.empty();
  }
}
