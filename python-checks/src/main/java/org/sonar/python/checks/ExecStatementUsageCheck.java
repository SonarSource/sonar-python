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
import org.sonar.plugins.python.api.tree.ExecStatement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = ExecStatementUsageCheck.CHECK_KEY)
public class ExecStatementUsageCheck extends PythonSubscriptionCheck {
  public static final String CHECK_KEY = "ExecStatementUsage";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.EXEC_STMT, ctx -> {
      ExecStatement tree = (ExecStatement) ctx.syntaxNode();
      ctx.addIssue(tree.execKeyword(), "Do not use exec statement.");
    });
  }
}
