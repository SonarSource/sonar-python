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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ReprExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "BackticksUsage")
public class BackticksUsageCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.REPR, ctx -> {
      ReprExpression node = (ReprExpression) ctx.syntaxNode();
      PreciseIssue issue = ctx.addIssue(node, "Use \"repr\" instead.");

      PythonTextEdit text1 = TextEditUtils
              .replace(node.openingBacktick(), "repr(");
      PythonTextEdit text2 = TextEditUtils
              .replace(node.closingBacktick(), ")");
      PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Replace backtick with \"repr()\".")
              .addTextEdit(text1)
              .addTextEdit(text2)
              .build();
      issue.addQuickFix(quickFix);
    });
  }
}
