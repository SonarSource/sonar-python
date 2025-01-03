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

@Rule(key = LineLengthCheck.CHECK_KEY)
public class LineLengthCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "LineLength";
  private static final int DEFAULT_MAXIMUM_LINE_LENGTH = 120;

  @RuleProperty(
    key = "maximumLineLength",
    description = "The maximum authorized line length",
    defaultValue = "" + DEFAULT_MAXIMUM_LINE_LENGTH)
  public int maximumLineLength = DEFAULT_MAXIMUM_LINE_LENGTH;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      String[] lines = ctx.pythonFile().content().split("\r?\n|\r", -1);
      for (int i = 0; i < lines.length; i++) {
        String line = lines[i];
        if (line.length() > maximumLineLength) {
          String message = MessageFormat.format("The line contains {0,number,integer} characters which is greater than {1,number,integer} authorized.",
            line.length(), maximumLineLength);
          ctx.addLineIssue(message, i + 1);
        }
      }
    });
  }
}
