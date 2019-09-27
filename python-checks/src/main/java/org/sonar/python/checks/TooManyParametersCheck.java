/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.LambdaExpression;
import org.sonar.python.api.tree.Tree.Kind;

@Rule(key = TooManyParametersCheck.CHECK_KEY)
public class TooManyParametersCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "S107";
  private static final String MESSAGE = "%s has %s parameters, which is greater than the %s authorized.";

  private static final int DEFAULT_MAX = 7;

  @RuleProperty(
    key = "max",
    defaultValue = "" + DEFAULT_MAX)
  public int max = DEFAULT_MAX;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> {
      FunctionDef tree = (FunctionDef) ctx.syntaxNode();
      if (tree.parameters() != null) {
        int nbParameters = tree.parameters().all().size();
        if (nbParameters > max) {
          String typeName = tree.isMethodDefinition() ? "Method" : "Function";
          String name = String.format("%s \"%s\"", typeName, tree.name().name());
          String message = String.format(MESSAGE, name, nbParameters, max);
          ctx.addIssue(tree.parameters(), message);
        }
      }
    });

    context.registerSyntaxNodeConsumer(Kind.LAMBDA, ctx -> {
      LambdaExpression tree = (LambdaExpression) ctx.syntaxNode();
      if (tree.parameters() != null) {
        int nbParameters = tree.parameters().all().size();
        if (nbParameters > max) {
          String name = "Lambda";
          String message = String.format(MESSAGE, name, nbParameters, max);
          ctx.addIssue(tree.parameters(), message);
        }
      }
    });
  }
}
