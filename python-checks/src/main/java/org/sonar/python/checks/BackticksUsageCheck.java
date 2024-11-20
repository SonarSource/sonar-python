/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ReprExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
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
