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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.IssueLocation;
import org.sonar.python.PythonCheckAstNode;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.metrics.CognitiveComplexityVisitor;

@Rule(key = CognitiveComplexityFunctionCheck.CHECK_KEY)
public class CognitiveComplexityFunctionCheck extends PythonCheckAstNode {

  private static final String MESSAGE = "Refactor this function to reduce its Cognitive Complexity from %s to the %s allowed.";
  public static final String CHECK_KEY = "S3776";
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
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    if (astNode.hasAncestor(PythonGrammar.FUNCDEF)) {
      return;
    }
    List<IssueLocation> secondaryLocations = new ArrayList<>();
    int complexity = CognitiveComplexityVisitor.complexity(astNode, (node, message) -> secondaryLocations.add(IssueLocation.preciseLocation(node, message)));
    if (complexity > threshold){
      String message = String.format(MESSAGE, complexity, threshold);
      PreciseIssue issue = addIssue(astNode.getFirstChild(PythonGrammar.FUNCNAME), message)
        .withCost(complexity - threshold);
      secondaryLocations.forEach(issue::secondary);
    }
  }

}
