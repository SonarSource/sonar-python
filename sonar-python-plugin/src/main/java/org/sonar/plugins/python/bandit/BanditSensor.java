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
package org.sonar.plugins.python.bandit;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Set;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.issue.NewExternalIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.config.Configuration;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.rules.RuleType;
import org.sonar.api.utils.Version;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.ExternalIssuesSensor;
import org.sonar.plugins.python.bandit.BanditJsonReportReader.Issue;
import org.sonarsource.analyzer.commons.internal.json.simple.parser.ParseException;

import static org.apache.commons.lang.StringUtils.isEmpty;

public class BanditSensor extends ExternalIssuesSensor {

  private static final Logger LOG = LoggerFactory.getLogger(BanditSensor.class);

  public static final String LINTER_NAME = "Bandit";
  public static final String LINTER_KEY = "bandit";
  public static final String REPORT_PATH_KEY = "sonar.python.bandit.reportPaths";

  private static final Long DEFAULT_CONSTANT_DEBT_MINUTES = 5L;

  @Override
  protected void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) throws IOException, ParseException {
    InputStream in = new FileInputStream(reportPath);
    LOG.info("Importing {}", reportPath);
    BanditJsonReportReader.read(in, issue -> saveIssue(context, issue, unresolvedInputFiles));
  }


  private static void saveIssue(SensorContext context, Issue issue, Set<String> unresolvedInputFiles) {
    if (isEmpty(issue.ruleKey) || isEmpty(issue.filePath) || isEmpty(issue.message)) {
      LOG.debug("Missing information for ruleKey:'{}', filePath:'{}', message:'{}'", issue.ruleKey, issue.filePath, issue.message);
      return;
    }

    InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().hasPath(issue.filePath));
    if (inputFile == null) {
      unresolvedInputFiles.add(issue.filePath);
      return;
    }

    NewExternalIssue newExternalIssue = context.newExternalIssue();
    newExternalIssue
      .type(RuleType.VULNERABILITY)
      .severity(toSonarQubeSeverity(issue.severity, issue.confidence))
      .remediationEffortMinutes(DEFAULT_CONSTANT_DEBT_MINUTES);

    NewIssueLocation primaryLocation = newExternalIssue.newLocation()
      .message(issue.message)
      .on(inputFile);

    if (issue.lineNumber != null) {
      primaryLocation.at(inputFile.selectLine(issue.lineNumber));
    }

    newExternalIssue.at(primaryLocation);
    newExternalIssue.engineId(LINTER_KEY);
    newExternalIssue.ruleId(issue.ruleKey).save();
  }

  private static Severity toSonarQubeSeverity(String severity, String confidence) {
    if ("HIGH".equalsIgnoreCase(severity)) {
      return "HIGH".equalsIgnoreCase(confidence) ? Severity.BLOCKER : Severity.CRITICAL;
    } else if ("MEDIUM".equalsIgnoreCase(severity)) {
      return Severity.MAJOR;
    } else {
      return Severity.MINOR;
    }
  }

  @Override
  protected boolean shouldExecute(Configuration conf) {
    return conf.hasKey(REPORT_PATH_KEY);
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
