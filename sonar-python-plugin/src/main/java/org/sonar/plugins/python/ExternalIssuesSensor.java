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
package org.sonar.plugins.python;


import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.issue.NewExternalIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.rules.RuleType;
import org.sonar.api.utils.log.Logger;
import java.util.stream.Collectors;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonarsource.analyzer.commons.ExternalReportProvider;

public abstract class ExternalIssuesSensor implements Sensor {

  private static final int MAX_LOGGED_FILE_NAMES = 20;
  private static final Long DEFAULT_CONSTANT_DEBT_MINUTES = 5L;

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyWhenConfiguration(conf -> conf.hasKey(reportPathKey()))
      .onlyOnLanguage(Python.KEY)
      .name("Import of " + linterName() + " issues");
  }

  @Override
  public void execute(SensorContext context) {
    Set<String> unresolvedInputFiles = new HashSet<>();
    List<File> reportFiles = ExternalReportProvider.getReportFiles(context, reportPathKey());
    reportFiles.forEach(report -> importReport(report, context, unresolvedInputFiles));
    logUnresolvedInputFiles(unresolvedInputFiles);
  }

  private void logUnresolvedInputFiles(Set<String> unresolvedInputFiles) {
    if (unresolvedInputFiles.isEmpty()) {
      return;
    }
    String fileList = unresolvedInputFiles.stream().sorted().limit(MAX_LOGGED_FILE_NAMES).collect(Collectors.joining(";"));
    if (unresolvedInputFiles.size() > MAX_LOGGED_FILE_NAMES) {
      fileList += ";...";
    }
    logger().warn("Failed to resolve {} file path(s) in " + linterName() + " report. No issues imported related to file(s): {}", unresolvedInputFiles.size(), fileList);
  }

  protected void saveIssue(SensorContext context, TextReportReader.Issue issue, Set<String> unresolvedInputFiles, String linterKey) {
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
    if (issue.columnNumber != null && issue.columnNumber < inputFile.selectLine(issue.lineNumber).end().lineOffset()) {
      primaryLocation.at(inputFile.newRange(issue.lineNumber, issue.columnNumber, issue.lineNumber, issue.columnNumber + 1));
    } else {
      // Pylint formatted issues might not provide column information
      primaryLocation.at(inputFile.selectLine(issue.lineNumber));
    }

    newExternalIssue.at(primaryLocation);
    newExternalIssue.engineId(linterKey).ruleId(issue.ruleKey);
    newExternalIssue.save();
  }

  protected abstract void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles);

  protected abstract String linterName();

  protected abstract String reportPathKey();

  protected abstract Logger logger();
}
