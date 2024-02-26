/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.metrics.CognitiveComplexityVisitor;

@Rule(key = CognitiveComplexityFunctionCheck.CHECK_KEY)
public class CognitiveComplexityFunctionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Refactor this function to reduce its Cognitive Complexity from %s to the %s allowed.";
  public static final String CHECK_KEY = "S3776";
  private static final int DEFAULT_THRESHOLD = 15;

  @RuleProperty(
    key = "threshold",
    description = "The maximum authorized complexity.",
    defaultValue = "" + DEFAULT_THRESHOLD)
  private int threshold = DEFAULT_THRESHOLD;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (isInnerFunction(functionDef)) {
        return;
      }
      List<IssueLocation> secondaryLocations = new ArrayList<>();
      int complexity = CognitiveComplexityVisitor.complexity(functionDef, (node, message) -> secondaryLocations.add(IssueLocation.preciseLocation(node, message)));
      if (complexity > threshold){
        String message = String.format(MESSAGE, complexity, threshold);
        PreciseIssue issue = ctx.addIssue(functionDef.name(), message)
          .withCost(complexity - threshold);
        secondaryLocations.forEach(issue::secondary);
      }
    });
  }

  private static boolean isInnerFunction(FunctionDef functionDef) {
    Tree parent = functionDef.parent();
    while (parent != null) {
      if (parent.is(Tree.Kind.FUNCDEF)) {
        return true;
      }
      parent = parent.parent();
    }
    return false;
  }

  public void setThreshold(int threshold) {
    this.threshold = threshold;
  }
}
