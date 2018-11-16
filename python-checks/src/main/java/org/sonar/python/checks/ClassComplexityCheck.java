/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
import java.util.Collections;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.metrics.ComplexityVisitor;

@Rule(key = "ClassComplexity")
public class ClassComplexityCheck extends PythonCheck {
  private static final int DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD = 200;
  private static final String MESSAGE = "Class has a complexity of %s which is greater than %s authorized.";

  @RuleProperty(key = "maximumClassComplexityThreshold", defaultValue = "" + DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD)
  int maximumClassComplexityThreshold = DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    int complexity = ComplexityVisitor.complexity(node);
    if (complexity > maximumClassComplexityThreshold) {
      String message = String.format(MESSAGE, complexity, maximumClassComplexityThreshold);
      addIssue(node.getFirstChild(PythonGrammar.CLASSNAME), message)
        .withCost(complexity - maximumClassComplexityThreshold);
    }
  }

}
