/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.plugins.python.ruff;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Set;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.Severity;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.issue.NewExternalIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.config.Configuration;
import org.sonar.api.rules.RuleType;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.ExternalIssuesSensor;
import org.sonar.plugins.python.TextReportReader;
import org.sonar.plugins.python.TextReportReader.Issue;
import org.sonarsource.analyzer.commons.internal.json.simple.parser.ParseException;

import static org.apache.commons.lang.StringUtils.isEmpty;

public class RuffSensor extends ExternalIssuesSensor {

  private static final Logger LOG = Loggers.get(RuffSensor.class);

  public static final String LINTER_NAME = "Ruff";
  public static final String LINTER_KEY = "ruff";
  public static final String REPORT_PATH_KEY = "sonar.python.ruff.reportPaths";

  private static final Long DEFAULT_CONSTANT_DEBT_MINUTES = 5L;

  @Override
  protected boolean shouldExecute(Configuration conf) {
    return conf.hasKey(REPORT_PATH_KEY);
  }

  @Override
  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  @Override
  protected String linterName() {
    return LINTER_NAME;
  }

  @Override
  protected Logger logger() {
    return LOG;
  }


  @Override
  protected void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) throws IOException, ParseException {
    InputStream in = new FileInputStream(reportPath);
    LOG.info("Importing {}", reportPath);
    RuffJsonReportReader.read(in, issue -> saveIssue(context, issue, unresolvedInputFiles));
  }


  private static void saveIssue(SensorContext context, RuffJsonReportReader.Issue issue, Set<String> unresolvedInputFiles) {
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
      .type(RuleType.CODE_SMELL)
      .severity(Severity.MAJOR)
      .remediationEffortMinutes(DEFAULT_CONSTANT_DEBT_MINUTES);

    NewIssueLocation primaryLocation = newExternalIssue.newLocation()
      .message(issue.message)
      .on(inputFile);

    if (issue.startLocationRow != null ){
      if(isValidEndLocation(issue, inputFile)) {
        primaryLocation.at(inputFile.newRange(issue.startLocationRow, issue.startLocationCol, issue.endLocationRow, issue.endLocationCol));
      }else {
        primaryLocation.at(inputFile.selectLine(issue.startLocationRow));
      }
    }

    newExternalIssue.at(primaryLocation);
    newExternalIssue.engineId(LINTER_KEY);
    newExternalIssue.ruleId(issue.ruleKey).save();
  }

  /*
   * The end location column should be after the start location col
   */
  private static boolean isValidEndLocation(RuffJsonReportReader.Issue issue, InputFile inputFile){
    return issue.startLocationCol != null && 
          issue.endLocationRow != null && 
          issue.endLocationCol != null &&
          isColInBounds(issue.startLocationRow, issue.startLocationCol, inputFile) && 
          (issue.endLocationRow != issue.startLocationRow || issue.endLocationCol != issue.startLocationCol);
  }

  private static boolean isColInBounds(int lineNumber, int columnNumber,InputFile inputFile){
    return columnNumber < inputFile.selectLine(lineNumber).end().lineOffset();
  }


}
