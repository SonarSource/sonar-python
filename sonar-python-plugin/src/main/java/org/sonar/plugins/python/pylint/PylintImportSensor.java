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
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.config.Settings;
import org.sonar.api.issue.Issuable;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonReportSensor;

import java.io.File;
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
public class PylintImportSensor extends PythonReportSensor  {
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPath";
  private static final String DEFAULT_REPORT_PATH = "pylint-reports/pylint-result-*.txt";

  private static final Logger LOG = LoggerFactory.getLogger(PylintImportSensor.class);

  private RuleFinder ruleFinder;
  private RulesProfile profile;
  private ResourcePerspectives resourcePerspectives;

  public PylintImportSensor(Settings conf, RuleFinder ruleFinder, RulesProfile profile, FileSystem fileSystem, ResourcePerspectives resourcePerspectives) {
    super(conf, fileSystem);

    this.ruleFinder = ruleFinder;
    this.profile = profile;
    this.resourcePerspectives = resourcePerspectives;
  }

  public boolean shouldExecuteOnProject(Project project) {
    FilePredicates p = fileSystem.predicates();
    boolean hasFiles = fileSystem.hasFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    boolean hasRules = !profile.getActiveRulesByRepository(PylintRuleRepository.REPOSITORY_KEY).isEmpty();
    return hasFiles && hasRules && conf.getString(REPORT_PATH_KEY) != null;
  }

  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  protected String defaultReportPath() {
    return DEFAULT_REPORT_PATH;
  }

  protected void processReports(final SensorContext context, List<File> reports)
      throws javax.xml.stream.XMLStreamException {
    List<Issue> issues = new LinkedList<Issue>();
    for(File report: reports){
      try {
        issues.addAll(parse(report));
      } catch(java.io.FileNotFoundException e) {
        LOG.error("Report '{}' cannot be found, details: '{}'", report, e.toString());
      }
    }

    saveIssues(issues, context);
  }

  private List<Issue> parse(File report) throws java.io.FileNotFoundException {
    List<Issue> issues = new LinkedList<Issue>();

    PylintReportParser parser = new PylintReportParser();
    for(Scanner sc = new Scanner(report); sc.hasNext(); ) {
      String line = sc.nextLine();
      Issue issue = parser.parseLine(line);
      if (issue != null){
        issues.add(issue);
      }
    }

    return issues;
  }

  private void saveIssues(List<Issue> issues, SensorContext context) {
    for (Issue pylintIssue : issues) {
      String filepath = pylintIssue.getFilename();
      InputFile pyfile = fileSystem.inputFile(fileSystem.predicates().hasPath(filepath));
      if(pyfile != null){
        Rule rule = ruleFinder.findByKey(PylintRuleRepository.REPOSITORY_KEY, pylintIssue.getRuleId());

        if (rule != null) {
          if (rule.isEnabled()) {
            Issuable issuable = resourcePerspectives.as(Issuable.class, pyfile);

            if (issuable != null) {
              org.sonar.api.issue.Issue issue = issuable.newIssueBuilder()
                .ruleKey(RuleKey.of(rule.getRepositoryKey(), rule.getKey()))
                .line(pylintIssue.getLine())
                .message(pylintIssue.getDescr())
                .build();
              issuable.addIssue(issue);
            }
          } else {
            LOG.debug("Pylint rule '{}' is disabled in Sonar", pylintIssue.getRuleId());
          }
        } else {
          LOG.warn("Pylint rule '{}' is unknown in Sonar", pylintIssue.getRuleId());
        }
      } else{
        LOG.warn("Cannot find the file '{}' in SonarQube, ignoring violation", filepath);
      }
    }
  }
}
