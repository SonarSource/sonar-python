/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonMetric;
import org.sonar.squidbridge.annotations.SqaleLinearWithOffsetRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.api.SourceClass;
import org.sonar.squidbridge.checks.ChecksHelper;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
    key = ClassComplexityCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Classes should not be too complex",
    tags = Tags.BRAIN_OVERLOAD
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.UNDERSTANDABILITY)
@SqaleLinearWithOffsetRemediation(
    coeff = "1min",
    offset = "10min",
    effortToFixDescription = "per complexity point over the threshold")
public class ClassComplexityCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "ClassComplexity";
  private static final int DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD = 200;

  @RuleProperty(key = "maximumClassComplexityThreshold", defaultValue = "" + DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD)
  private int maximumClassComplexityThreshold = DEFAULT_MAXIMUM_CLASS_COMPLEXITY_THRESHOLD;

  @Override
  public void init() {
    subscribeTo(PythonGrammar.CLASSDEF);
  }

  @Override
  public void leaveNode(AstNode node) {
    SourceClass sourceClass = (SourceClass) getContext().peekSourceCode();
    int complexity = ChecksHelper.getRecursiveMeasureInt(sourceClass, PythonMetric.COMPLEXITY);
    if (complexity > maximumClassComplexityThreshold) {
      getContext().createLineViolation(this,
          "Class has a complexity of {0,number,integer} which is greater than {1,number,integer} authorized.",
          node,
          complexity,
          maximumClassComplexityThreshold);
    }
  }

  public void setMaximumClassComplexityThreshold(int threshold) {
    this.maximumClassComplexityThreshold = threshold;
  }

}
