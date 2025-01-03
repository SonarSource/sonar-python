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
package org.sonar.python.checks;

import java.text.MessageFormat;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = TooManyLinesInFileCheck.CHECK_KEY)
public class TooManyLinesInFileCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S104";
  private static final int DEFAULT = 1000;
  private static final String MESSAGE = "File \"{0}\" has {1} lines, which is greater than {2} authorized. Split it into smaller files.";

  @RuleProperty(
    key = "maximum",
    description = "Maximum authorized lines in a file",
    defaultValue = "" + DEFAULT)
  public int maximum = DEFAULT;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      int line = ctx.syntaxNode().lastToken().line();
      if (line > maximum) {
        ctx.addFileIssue(MessageFormat.format(MESSAGE, ctx.pythonFile().fileName(), line, maximum));
      }
    });
  }
}

