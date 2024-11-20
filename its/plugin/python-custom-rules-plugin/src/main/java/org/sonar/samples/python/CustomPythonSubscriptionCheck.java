/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package org.sonar.samples.python;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(
  key = CustomPythonSubscriptionCheck.RULE_KEY_SUBSCRIPTION,
  priority = Priority.MINOR,
  name = "Python subscription visitor check",
  description = "desc")
public class CustomPythonSubscriptionCheck extends PythonSubscriptionCheck {

  public static final String RULE_KEY_SUBSCRIPTION = "subscription";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, ctx -> ctx.addIssue(((ForStatement) ctx.syntaxNode()).forKeyword(), "For statement."));
  }
}
