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
import java.util.Optional;
import java.util.Set;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.server.debt.DebtRemediationFunction;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.ExternalIssuesSensor;
import org.sonar.plugins.python.TextReportReader;
import org.sonar.plugins.python.TextReportReader.Issue;

public class PylintSensor extends ExternalIssuesSensor {

  private static final Logger LOG = Loggers.get(PylintSensor.class);

  public static final String LINTER_NAME = "Pylint";
  public static final String LINTER_KEY = "pylint";
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPaths";
  private static final RulesDefinition.Repository ruleRepository;

  static {
    RulesDefinition.Context ruleDefinitionContext = new RulesDefinition.Context();
    PylintRulesDefinition rulesDefinition = new PylintRulesDefinition();
    rulesDefinition.define(ruleDefinitionContext);
    ruleRepository = ruleDefinitionContext.repository("external_pylint");
  }

  @Override
  protected void importReport(File reportPath, SensorContext context, Set<String> unresolvedInputFiles) throws IOException {
    List<Issue> issues = new TextReportReader(TextReportReader.COLUMN_ZERO_BASED).parse(reportPath, context.fileSystem());
    issues.forEach(issue -> {
      Optional.ofNullable(ruleRepository)
        .map(r -> r.rule(issue.ruleKey))
        .map(RulesDefinition.Rule::debtRemediationFunction)
        .map(DebtRemediationFunction::baseEffort)
        .map(b -> Long.valueOf(b.split("[^0-9]+")[0]))
        .ifPresent(issue::setDebRemediationEffort);
      saveIssue(context, issue, unresolvedInputFiles, LINTER_KEY);
    });
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
