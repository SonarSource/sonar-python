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
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.metrics.ComplexityVisitor;

@Rule(key = "ClassComplexity")
public class ClassComplexityCheck extends PythonSubscriptionCheck {
  private static final int DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD = 200;
  private static final String MESSAGE = "Class has a complexity of %s which is greater than %s authorized.";

  @RuleProperty(
    key = "maximumClassComplexityThreshold",
    description = "Maximum complexity allowed",
    defaultValue = "" + DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD)
  int maximumClassComplexityThreshold = DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      int complexity = ComplexityVisitor.complexity(classDef);
      if (complexity > maximumClassComplexityThreshold) {
        String message = String.format(MESSAGE, complexity, maximumClassComplexityThreshold);
        ctx.addIssue(classDef.name(), message).withCost(complexity - maximumClassComplexityThreshold);
      }
    });
  }
}
