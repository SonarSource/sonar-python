/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import com.intellij.lang.ASTNode;
import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyFunction;
import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheck;
import org.sonar.python.metrics.CognitiveComplexityVisitor;

@Rule(key = "S3776")
public class CognitiveComplexityFunctionCheck extends PythonCheck {

  private static final String MESSAGE = "Refactor this function to reduce its Cognitive Complexity from %s to the %s allowed.";
  private static final int DEFAULT_THRESHOLD = 15;

  @RuleProperty(
    key = "threshold",
    description = "The maximum authorized complexity.",
    defaultValue = "" + DEFAULT_THRESHOLD)
  private int threshold = DEFAULT_THRESHOLD;

  public void setThreshold(int threshold) {
    this.threshold = threshold;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.FUNCTION_DECLARATION, ctx -> {
      PyFunction function = (PyFunction) ctx.syntaxNode();
      if (PsiTreeUtil.getParentOfType(function, PyFunction.class) != null) {
        return;
      }
      List<IssueLocation> secondaryLocations = new ArrayList<>();
      int complexity = CognitiveComplexityVisitor.complexity(function, (node, message) -> secondaryLocations.add(IssueLocation.preciseLocation(node, message)));
      if (complexity > threshold){
        String message = String.format(MESSAGE, complexity, threshold);
        ASTNode nameNode = function.getNameNode();
        if (nameNode != null) {
          PreciseIssue issue = ctx.addIssue(nameNode.getPsi(), message)
            .withCost(complexity - threshold);
          secondaryLocations.forEach(issue::secondary);
        }
      }
    });
  }

}
