/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.Sensor;
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

import java.io.File;
import java.io.IOException;
import java.util.List;

public class PylintSensor implements Sensor {
  public static final String REPORT_PATH_KEY = "sonar.python.pylint.reportPath";

  private static final Logger LOG = LoggerFactory.getLogger(PylintSensor.class);

  private ActiveRules activeRules;
  private PylintConfiguration conf;
  private FileSystem fileSystem;
  private ResourcePerspectives resourcePerspectives;
  private Settings settings;


  public PylintSensor(PylintConfiguration conf, ActiveRules activeRules, FileSystem fileSystem, ResourcePerspectives resourcePerspectives, Settings settings) {
    this.conf = conf;
    this.activeRules = activeRules;
    this.fileSystem = fileSystem;
    this.resourcePerspectives = resourcePerspectives;
    this.settings = settings;
  }

  @Override
  public boolean shouldExecuteOnProject(Project project) {
    FilePredicates p = fileSystem.predicates();
    boolean hasFiles = fileSystem.hasFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    boolean hasRules = !activeRules.findByRepository(PylintRuleRepository.REPOSITORY_KEY).isEmpty();
    return hasFiles && hasRules && settings.getString(REPORT_PATH_KEY) == null;
  }

  @Override
  public void analyse(Project project, SensorContext sensorContext) {
    File workDir = new File(fileSystem.workDir(), "/pylint/");
    prepareWorkDir(workDir);
    int i = 0;
    FilePredicates p = fileSystem.predicates();
    Iterable<File> files = fileSystem.files(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    for (File file : files) {
      try {
        File out = new File(workDir, i + ".out");
        analyzeFile(file, out);
        i++;
      } catch (Exception e) {
        String msg = new StringBuilder()
            .append("Cannot analyse the file '")
            .append(file.getAbsolutePath())
            .append("', details: '")
            .append(e)
            .append("'")
            .toString();
        throw new IllegalStateException(msg, e);
      }
    }
  }

  protected void analyzeFile(File file, File out) throws IOException {
    InputFile pyFile = fileSystem.inputFile(fileSystem.predicates().is(file));

    String pylintConfigPath = conf.getPylintConfigPath(fileSystem);
    String pylintPath = conf.getPylintPath();

    PylintIssuesAnalyzer analyzer = new PylintIssuesAnalyzer(pylintPath, pylintConfigPath);
    List<Issue> issues = analyzer.analyze(file.getAbsolutePath(), fileSystem.encoding(), out);

    for (Issue pylintIssue : issues) {
      ActiveRule rule = activeRules.find(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, pylintIssue.getRuleId()));

      if (rule != null) {
        Issuable issuable = resourcePerspectives.as(Issuable.class, pyFile);
        if (issuable != null) {
          org.sonar.api.issue.Issue issue = issuable.newIssueBuilder()
              .ruleKey(rule.ruleKey())
              .line(pylintIssue.getLine())
              .message(pylintIssue.getDescription())
              .build();
          issuable.addIssue(issue);
        }
      } else {
        LOG.warn("Pylint rule '{}' is unknown in Sonar", pylintIssue.getRuleId());
      }
    }
  }

  private static void prepareWorkDir(File dir) {
    try {
      FileUtils.forceMkdir(dir);
      // directory is cleaned, because Sonar 3.0 will not do this for us
      FileUtils.cleanDirectory(dir);
    } catch (IOException e) {
      throw new IllegalStateException("Cannot create directory: " + dir, e);
    }
  }

}
