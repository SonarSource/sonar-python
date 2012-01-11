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
import org.sonar.api.resources.InputFile;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.api.resources.Resource;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.rules.Violation;

public class PythonViolationsSensor implements Sensor {

  private static final Logger LOGGER = LoggerFactory.getLogger(PythonViolationsSensor.class);

  private RuleFinder ruleFinder;

  public PythonViolationsSensor(RuleFinder ruleFinder) {
    this.ruleFinder = ruleFinder;
  }

  public void analyse(Project project, SensorContext sensorContext) {
    for (InputFile inputFile : project.getFileSystem().mainFiles(Python.KEY)) {
      try {
        analyzeFile(inputFile, project.getFileSystem(), sensorContext);
      } catch (Exception e) {
        LOGGER.error("Cannot analyze the file '{}', details: '{}'", inputFile.getFile().getAbsolutePath(), e);
      }
    }
  }

  protected void analyzeFile(InputFile inputFile, ProjectFileSystem projectFileSystem, SensorContext sensorContext) throws IOException {
    Resource pyfile = PythonFile.fromIOFile(inputFile.getFile(), projectFileSystem.getSourceDirs());
    List<Issue> issues = new PythonViolationsAnalyzer().analyze(inputFile.getFile().getPath());
    for (Issue issue : issues) {
      Rule rule = ruleFinder.findByKey(PythonRuleRepository.REPOSITORY_KEY, issue.ruleId);
      Violation violation = Violation.create(rule, pyfile);
      violation.setLineId(issue.line);
      violation.setMessage(issue.descr);
      sensorContext.saveViolation(violation);
    }
  }

  public boolean shouldExecuteOnProject(Project project) {
    return project.getLanguage().equals(Python.INSTANCE);
  }
}
