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
package org.sonar.plugins.python.pylint;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.ExternalIssuesSensor;
import org.sonar.plugins.python.TextReportReader;
import org.sonar.plugins.python.TextReportReader.Issue;

public class PylintSensor extends ExternalIssuesSensor {

  private static final Logger LOG = LoggerFactory.getLogger(PylintSensor.class);

  public static final String LINTER_NAME = "Pylint";
  public static final String LINTER_KEY = "pylint";
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPaths";

  @Override
  protected void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) throws IOException {
    List<Issue> issues = new TextReportReader(TextReportReader.COLUMN_ZERO_BASED).parse(reportPath, context.fileSystem());
    issues.forEach(i -> saveIssue(context, i, unresolvedInputFiles, LINTER_KEY));
  }

  @Override
  protected boolean shouldExecute(Configuration conf) {
    return conf.hasKey(REPORT_PATH_KEY) || conf.hasKey(PYLINT_LEGACY_KEY);
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
}
