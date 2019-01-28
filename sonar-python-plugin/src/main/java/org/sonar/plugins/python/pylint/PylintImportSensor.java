/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.ActiveRule;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.config.Configuration;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.PythonReportSensor;

public class PylintImportSensor extends PythonReportSensor {
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPath";
  private static final String DEFAULT_REPORT_PATH = "pylint-reports/pylint-result-*.txt";

  private static final Logger LOG = Loggers.get(PylintImportSensor.class);
  private static final Set<String> warningAlreadyLogged = new HashSet<>();
  private PylintRuleRepository pylintRuleRepository;

  public PylintImportSensor(Configuration conf, PylintRuleRepository pylintRuleRepository) {
    super(conf);
    this.pylintRuleRepository = pylintRuleRepository;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    super.describe(descriptor);
    descriptor
      .createIssuesForRuleRepository(PylintRuleRepository.REPOSITORY_KEY)
      .requireProperty(REPORT_PATH_KEY);
  }

  @Override
  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  @Override
  protected String defaultReportPath() {
    return DEFAULT_REPORT_PATH;
  }

  @Override
  protected void processReports(final SensorContext context, List<File> reports) {
    List<Issue> issues = new LinkedList<>();
    for (File report : reports) {
      try {
        issues.addAll(parse(report, context.fileSystem()));
      } catch (java.io.FileNotFoundException e) {
        LOG.error("Report '{}' cannot be found, details: '{}'", report, e);
      } catch (IOException e) {
        LOG.error("Report '{}' cannot be read, details: '{}'", report, e);
      }
    }

    saveIssues(issues, context);
  }

  private static List<Issue> parse(File report, FileSystem fileSystem) throws IOException {
    List<Issue> issues = new LinkedList<>();

    PylintReportParser parser = new PylintReportParser();
    Scanner sc;
    for (sc = new Scanner(report.toPath(), fileSystem.encoding().name()); sc.hasNext(); ) {
      String line = sc.nextLine();
      Issue issue = parser.parseLine(line);
      if (issue != null) {
        issues.add(issue);
      }
    }
    sc.close();
    return issues;
  }

  private void saveIssues(List<Issue> issues, SensorContext context) {
    FileSystem fileSystem = context.fileSystem();
    for (Issue pylintIssue : issues) {
      String filepath = pylintIssue.getFilename();
      InputFile pyfile = fileSystem.inputFile(fileSystem.predicates().hasPath(filepath));
      if (pyfile != null) {
        ActiveRule rule = context.activeRules().find(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, pylintIssue.getRuleId()));
        processRule(pylintIssue, pyfile, rule, context, pylintRuleRepository);
      } else {
        LOG.warn("Cannot find the file '{}' in SonarQube, ignoring violation", filepath);
      }
    }
  }

  public static void processRule(Issue pylintIssue, InputFile pyfile, @Nullable ActiveRule rule, SensorContext context, PylintRuleRepository pylintRuleRepository) {
    if (rule != null) {
      NewIssue newIssue = context
        .newIssue()
        .forRule(rule.ruleKey());
      newIssue.at(
        newIssue.newLocation()
          .on(pyfile)
          .at(pyfile.selectLine(pylintIssue.getLine()))
          .message(pylintIssue.getDescription()));
      newIssue.save();
    } else if (!pylintRuleRepository.hasRuleDefinition(pylintIssue.getRuleId())) {
      logUnknownRuleWarning(pylintIssue.getRuleId());
    }
  }

  private static void logUnknownRuleWarning(String ruleId) {
    if (!warningAlreadyLogged.contains(ruleId)) {
      warningAlreadyLogged.add(ruleId);
      LOG.warn("Pylint rule '{}' is unknown in Sonar", ruleId);
    }
  }

}
