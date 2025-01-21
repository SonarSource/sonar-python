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
package org.sonar.plugins.python;


import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.batch.sensor.issue.NewExternalIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.config.Configuration;
import org.sonar.api.rules.RuleType;
import org.slf4j.Logger;
import org.sonar.plugins.python.pylint.PylintSensor;
import org.sonarsource.analyzer.commons.ExternalReportProvider;
import org.sonarsource.analyzer.commons.internal.json.simple.parser.ParseException;

public abstract class ExternalIssuesSensor implements Sensor {

  private static final int MAX_LOGGED_FILE_NAMES = 20;
  private static final Long DEFAULT_CONSTANT_DEBT_MINUTES = 5L;
  protected static final String PYLINT_LEGACY_KEY = "sonar.python.pylint.reportPath";

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyWhenConfiguration(this::shouldExecute)
      .onlyOnLanguage(Python.KEY)
      .name("Import of " + linterName() + " issues");
  }

  @Override
  public void execute(SensorContext context) {
    Set<String> unresolvedInputFiles = new HashSet<>();
    List<File> reportFiles = ExternalReportProvider.getReportFiles(context, reportPathKey());
    if (reportFiles.isEmpty() && context.config().hasKey(PYLINT_LEGACY_KEY)) {
      reportFiles = ExternalReportProvider.getReportFiles(context, PYLINT_LEGACY_KEY);
      logger().warn("The use of '{}' is deprecated. Please use the '{}' property instead.", PYLINT_LEGACY_KEY, PylintSensor.REPORT_PATH_KEY);
    }
    reportFiles.forEach(report -> importExternalReport(report, context, unresolvedInputFiles));
    logUnresolvedInputFiles(unresolvedInputFiles);
  }

  private void importExternalReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) {
    try {
      importReport(reportPath, context, unresolvedInputFiles);
    } catch (IOException | ParseException | RuntimeException e) {
      logFileCantBeRead(e, reportPath);
    }
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

  private void logFileCantBeRead(Exception e, File reportPath) {
    logger().error("No issues information will be saved as the report file '{}' can't be read. {}: {}"
      , reportPath, e.getClass().getSimpleName(), e.getMessage());
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

  protected abstract void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) throws IOException, ParseException;

  protected abstract boolean shouldExecute(Configuration conf);

  protected abstract String linterName();

  protected abstract String reportPathKey();

  protected abstract Logger logger();
}
