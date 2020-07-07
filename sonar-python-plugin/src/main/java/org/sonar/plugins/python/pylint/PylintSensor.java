/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.issue.NewExternalIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.rules.RuleType;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.ExternalIssuesSensor;

public class PylintSensor extends ExternalIssuesSensor {

  private static final Logger LOG = Loggers.get(PylintSensor.class);

  public static final String LINTER_NAME = "Pylint";
  public static final String LINTER_KEY = "pylint";
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPaths";

  private static final Long DEFAULT_CONSTANT_DEBT_MINUTES = 5L;

  @Override
  protected void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) {
    try {
      List<PylintReportReader.Issue> issues = new PylintReportReader().parse(reportPath, context.fileSystem());
      issues.forEach(i -> saveIssue(context, i, unresolvedInputFiles));
    } catch (IOException e) {
      LOG.error("No issues information will be saved as the report file '{}' can't be read. " +
        e.getClass().getSimpleName() + ": " + e.getMessage(), reportPath, e);
    }
  }

  private static void saveIssue(SensorContext context, PylintReportReader.Issue issue, Set<String> unresolvedInputFiles) {
    InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().hasPath(issue.filePath));
    if (inputFile == null) {
      unresolvedInputFiles.add(issue.filePath);
      return;
    }

    NewExternalIssue newExternalIssue = context.newExternalIssue();
    newExternalIssue
      .type(RuleType.CODE_SMELL)
      .severity(Severity.MAJOR)
      .remediationEffortMinutes(DEFAULT_CONSTANT_DEBT_MINUTES);

    NewIssueLocation primaryLocation = newExternalIssue.newLocation()
      .message(issue.message)
      .on(inputFile);
    if (issue.columnNumber != null && issue.columnNumber < inputFile.selectLine(issue.lineNumber).end().lineOffset() + 1) {
      inputFile.selectLine(issue.lineNumber).end().lineOffset();
      primaryLocation.at(inputFile.newRange(issue.lineNumber, issue.columnNumber, issue.lineNumber, issue.columnNumber + 1));
    } else {
      // Pylint formatted issues don't provide column information
      primaryLocation.at(inputFile.selectLine(issue.lineNumber));
    }

    newExternalIssue.at(primaryLocation);
    newExternalIssue.engineId(LINTER_KEY).ruleId(issue.ruleKey);
    newExternalIssue.save();
  }

  @Override
  protected String linterName() {
    return LINTER_NAME;
  }

  @Override
  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  @Override
  protected Logger logger() {
    return LOG;
  }
}
