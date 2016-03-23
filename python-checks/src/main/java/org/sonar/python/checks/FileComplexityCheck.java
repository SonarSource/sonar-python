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
import org.sonar.python.api.PythonMetric;
import org.sonar.squidbridge.annotations.SqaleLinearWithOffsetRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.ChecksHelper;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
    key = FileComplexityCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Files should not be too complex",
    tags = Tags.BRAIN_OVERLOAD
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.UNIT_TESTABILITY)
@SqaleLinearWithOffsetRemediation(
    coeff = "1min",
    offset = "30min",
    effortToFixDescription = "per complexity point above the threshold")
public class FileComplexityCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "FileComplexity";
  private static final int DEFAULT_MAXIMUM_FILE_COMPLEXITY_THRESHOLD = 200;

  @RuleProperty(
    key = "maximumFileComplexityThreshold",
    defaultValue = "" + DEFAULT_MAXIMUM_FILE_COMPLEXITY_THRESHOLD)
  private int maximumFileComplexityThreshold = DEFAULT_MAXIMUM_FILE_COMPLEXITY_THRESHOLD;

  @Override
  public void leaveFile(AstNode astNode) {
    SourceFile sourceFile = (SourceFile) getContext().peekSourceCode();
    int complexity = ChecksHelper.getRecursiveMeasureInt(sourceFile, PythonMetric.COMPLEXITY);
    if (complexity > maximumFileComplexityThreshold) {
      getContext().createFileViolation(this,
          "File has a complexity of {0,number,integer} which is greater than {1,number,integer} authorized.",
          complexity,
          maximumFileComplexityThreshold);
    }
  }

  public void setMaximumFileComplexityThreshold(int threshold) {
    this.maximumFileComplexityThreshold = threshold;
  }

}
