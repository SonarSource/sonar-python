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

import com.sonar.sslr.api.Token;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.Tree;

@Rule(key = "InequalityUsage")
public class InequalityUsageCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, ctx -> {
      PyBinaryExpressionTree expr = (PyBinaryExpressionTree) ctx.syntaxNode();
      Token operator = expr.operator();
      if (operator.getValue().equals(PythonPunctuator.NOT_EQU2.getValue())) {
        ctx.addIssue(operator, "Replace \"<>\" by \"!=\".");
      }
    });
  }
}
