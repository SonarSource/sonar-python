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
import org.sonar.python.metrics.ComplexityVisitor;

@Rule(key = "FileComplexity")
public class FileComplexityCheck extends PythonSubscriptionCheck {
  private static final int DEFAULT_MAXIMUM_FILE_COMPLEXITY_THRESHOLD = 200;

  @RuleProperty(
    key = "maximumFileComplexityThreshold",
    description = "The maximum authorized complexity in file",
    defaultValue = "" + DEFAULT_MAXIMUM_FILE_COMPLEXITY_THRESHOLD)
  int maximumFileComplexityThreshold = DEFAULT_MAXIMUM_FILE_COMPLEXITY_THRESHOLD;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      int complexity = ComplexityVisitor.complexity(ctx.syntaxNode());
      if (complexity > maximumFileComplexityThreshold) {
        String message = MessageFormat.format(
          "File has a complexity of {0,number,integer} which is greater than {1,number,integer} authorized.",
          complexity,
          maximumFileComplexityThreshold);
        ctx.addFileIssue(message).withCost(complexity - maximumFileComplexityThreshold);
      }
    });
  }
}
