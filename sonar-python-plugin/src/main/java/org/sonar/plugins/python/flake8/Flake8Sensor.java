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
package org.sonar.plugins.python.flake8;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.issue.Issuable;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.scan.filesystem.FileQuery;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.api.utils.SonarException;
import org.sonar.plugins.python.Python;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Flake8Sensor implements Sensor {

  private static final Logger LOG = LoggerFactory.getLogger(Flake8Sensor.class);

  private RuleFinder ruleFinder;
  private RulesProfile profile;
  private Flake8Configuration conf;
  private ModuleFileSystem fileSystem;
  private ResourcePerspectives resourcePerspectives;


  public Flake8Sensor(RuleFinder ruleFinder, Flake8Configuration conf, RulesProfile profile, ModuleFileSystem fileSystem, ResourcePerspectives resourcePerspectives) {
    this.ruleFinder = ruleFinder;
    this.conf = conf;
    this.profile = profile;
    this.fileSystem = fileSystem;
    this.resourcePerspectives = resourcePerspectives;
  }

  public boolean shouldExecuteOnProject(Project project) {
    return !fileSystem.files(FileQuery.onSource().onLanguage(Python.KEY)).isEmpty()
        && !profile.getActiveRulesByRepository(PylintRuleRepository.REPOSITORY_KEY).isEmpty();
  }

  public void analyse(Project project, SensorContext sensorContext) {
    File workdir = new File(fileSystem.workingDir(), "/flake8/");
    prepareWorkDir(workdir);
    int i = 0;
    for (File file : fileSystem.files(FileQuery.onSource().onLanguage(Python.KEY))) {
      try {
        File out = new File(workdir, i + ".out");
        analyzeFile(file, out, project, sensorContext);
        i++;
      } catch (Exception e) {
        String msg = new StringBuilder()
            .append("Cannot analyse the file '")
            .append(file.getAbsolutePath())
            .append("', details: '")
            .append(e)
            .append("'")
            .toString();
        throw new SonarException(msg, e);
      }
    }
  }

  protected void analyzeFile(File file, File out, Project project, SensorContext sensorContext) throws IOException {
    org.sonar.api.resources.File pyfile = org.sonar.api.resources.File.fromIOFile(file, project);

    String flake8ConfigPath = conf.getPylintConfigPath(fileSystem);
    String flake8Path = conf.getPylintPath();

    Flake8IssuesAnalyzer analyzer = new Flake8IssuesAnalyzer(flake8Path, flake8ConfigPath);
    List<Issue> issues = analyzer.analyze(file.getAbsolutePath(), fileSystem.sourceCharset(), out);

    for (Issue flake8Issue : issues) {
      Rule rule = ruleFinder.findByKey(PylintRuleRepository.REPOSITORY_KEY, flake8Issue.getRuleId());

      if (rule != null) {
        if (rule.isEnabled()) {
          Issuable issuable = resourcePerspectives.as(Issuable.class, pyfile);

          if (issuable != null) {
            org.sonar.api.issue.Issue issue = issuable.newIssueBuilder()
              .ruleKey(RuleKey.of(rule.getRepositoryKey(), rule.getKey()))
              .line(flake8Issue.getLine())
              .message(flake8Issue.getDescr())
              .build();
            issuable.addIssue(issue);
          }
        } else {
          LOG.debug("Pylint rule '{}' is disabled in Sonar", flake8Issue.getRuleId());
        }
      } else {
        LOG.warn("Pylint rule '{}' is unknown in Sonar", flake8Issue.getRuleId());
      }
    }
  }

  private static void prepareWorkDir(File dir) {
    try {
      FileUtils.forceMkdir(dir);
      // directory is cleaned, because Sonar 3.0 will not do this for us
      FileUtils.cleanDirectory(dir);
    } catch (IOException e) {
      throw new SonarException("Cannot create directory: " + dir, e);
    }
  }

}
