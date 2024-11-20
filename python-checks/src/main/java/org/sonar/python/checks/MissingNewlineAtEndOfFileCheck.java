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
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S113")
public class MissingNewlineAtEndOfFileCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Add a new line at the end of this file \"%s\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      String fileContent = ctx.pythonFile().content();
      if (fileContent.length() > 0 && !fileContent.endsWith("\n") && !fileContent.endsWith("\r")) {
        ctx.addFileIssue(String.format(MESSAGE, ctx.pythonFile().fileName()));
      }
    });
  }
}
