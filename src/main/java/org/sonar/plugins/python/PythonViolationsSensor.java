/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.api.resources.Resource;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.rules.Violation;

public class PythonViolationsSensor implements Sensor {

  private static final Logger LOGGER = LoggerFactory.getLogger(PythonViolationsSensor.class);

  private RuleFinder ruleFinder;
  private Python python;

  public PythonViolationsSensor(Python python, RuleFinder ruleFinder) {
    this.python = python;
    this.ruleFinder = ruleFinder;
  }

  public void analyse(Project project, SensorContext sensorContext) {
    for (File pythonFile : project.getFileSystem().getSourceFiles(python)) {
      try {
        analyzeFile(pythonFile, project.getFileSystem(), sensorContext);
      } catch (Exception e) {
        LOGGER.error("Cannot analyze the file '{}', details: '{}'", pythonFile.getAbsolutePath(), e);
      }
    }
  }

  protected void analyzeFile(File file, ProjectFileSystem projectFileSystem, SensorContext sensorContext) throws IOException {
    Resource pyfile = PythonFile.fromIOFile(file, projectFileSystem.getSourceDirs());
    List<Issue> issues = new PythonViolationsAnalyzer().analyze(file.getPath());
    for (Issue issue : issues) {
      Rule rule = ruleFinder.findByKey(PythonRuleRepository.REPOSITORY_KEY, issue.ruleId);
      Violation violation = Violation.create(rule, pyfile);
      violation.setLineId(issue.line);
      violation.setMessage(issue.descr);
      sensorContext.saveViolation(violation);
    }
  }

  public boolean shouldExecuteOnProject(Project project) {
    return project.getLanguage().equals(python);
  }
}
