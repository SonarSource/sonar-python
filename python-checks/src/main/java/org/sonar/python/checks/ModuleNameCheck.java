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

import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1578")
public class ModuleNameCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT = "(([a-z_][a-z0-9_]*)|([A-Z][a-zA-Z0-9]+))$";
  private static final String MESSAGE = "Rename this module to match this regular expression: \"%s\".";

  @RuleProperty(
    key = "format",
    description = "Regular expression used to check the module names against.",
    defaultValue = "" + DEFAULT)
  public String format = DEFAULT;

  private Pattern pattern = null;

  @Override
  public void initialize(SubscriptionCheck.Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      String fileName = ctx.pythonFile().fileName();
      int dotIndex = fileName.lastIndexOf('.');
      if (dotIndex > 0) {
        String moduleName = fileName.substring(0, dotIndex);
        if (!pattern().matcher(moduleName).matches()) {
          ctx.addFileIssue(String.format(MESSAGE, format));
        }
      }
    });
  }

  private Pattern pattern() {
    if (pattern == null) {
      pattern = Pattern.compile(format);
    }
    return pattern;
  }

}
