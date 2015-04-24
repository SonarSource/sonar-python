/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.plugins.python.pylint;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.Properties;
import org.sonar.api.Property;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.ActiveRule;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.config.Settings;
import org.sonar.api.issue.Issuable;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonReportSensor;

import javax.annotation.Nullable;
import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

@Properties({
    @Property(
        key = PylintImportSensor.REPORT_PATH_KEY,
        defaultValue = "",
        name = "Pylint's reports",
        description = "Path to Pylint's report file, relative to projects root",
        global = false,
        project = true)
})
public class PylintImportSensor extends PythonReportSensor {
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPath";
  private static final String DEFAULT_REPORT_PATH = "pylint-reports/pylint-result-*.txt";

  private static final Logger LOG = LoggerFactory.getLogger(PylintImportSensor.class);

  private ActiveRules activeRules;
  private ResourcePerspectives resourcePerspectives;

  public PylintImportSensor(Settings conf, ActiveRules activeRules, FileSystem fileSystem, ResourcePerspectives resourcePerspectives) {
    super(conf, fileSystem);

    this.activeRules = activeRules;
    this.resourcePerspectives = resourcePerspectives;
  }

  @Override
  public boolean shouldExecuteOnProject(Project project) {
    FilePredicates p = fileSystem.predicates();
    boolean hasFiles = fileSystem.hasFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    boolean hasRules = !activeRules.findByRepository(PylintRuleRepository.REPOSITORY_KEY).isEmpty();
    return hasFiles && hasRules && conf.getString(REPORT_PATH_KEY) != null;
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
  protected void processReports(final SensorContext context, List<File> reports)
      throws javax.xml.stream.XMLStreamException {
    List<Issue> issues = new LinkedList<>();
    for (File report : reports) {
      try {
        issues.addAll(parse(report));
      } catch (java.io.FileNotFoundException e) {
        LOG.error("Report '{}' cannot be found, details: '{}'", report, e);
      } catch (java.io.IOException e) {
        LOG.error("Report '{}' cannot be read, details: '{}'", report, e);
      }
    }

    saveIssues(issues);
  }

  private List<Issue> parse(File report) throws IOException {
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

  private void saveIssues(List<Issue> issues) {
    for (Issue pylintIssue : issues) {
      String filepath = pylintIssue.getFilename();
      InputFile pyfile = fileSystem.inputFile(fileSystem.predicates().hasPath(filepath));
      if (pyfile != null) {
        ActiveRule rule = activeRules.find(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, pylintIssue.getRuleId()));
        processRule(pylintIssue, pyfile, rule);
      } else {
        LOG.warn("Cannot find the file '{}' in SonarQube, ignoring violation", filepath);
      }
    }
  }

  private void processRule(Issue pylintIssue, InputFile pyfile, @Nullable ActiveRule rule) {
    if (rule != null) {
      Issuable issuable = resourcePerspectives.as(Issuable.class, pyfile);
      addIssue(pylintIssue, rule, issuable);
    } else {
      LOG.warn("Pylint rule '{}' is unknown in Sonar", pylintIssue.getRuleId());
    }
  }

  private void addIssue(Issue pylintIssue, ActiveRule rule, @Nullable Issuable issuable) {
    if (issuable != null) {
      org.sonar.api.issue.Issue issue = issuable.newIssueBuilder()
          .ruleKey(rule.ruleKey())
          .line(pylintIssue.getLine())
          .message(pylintIssue.getDescription())
          .build();
      issuable.addIssue(issue);
    }
  }
}
