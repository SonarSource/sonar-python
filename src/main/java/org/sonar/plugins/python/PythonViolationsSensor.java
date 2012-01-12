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

import java.io.IOException;
import java.util.List;

import org.apache.commons.configuration.Configuration;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.resources.InputFile;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.api.resources.Resource;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.rules.Violation;

public class PythonViolationsSensor extends PythonSensor {
  private RuleFinder ruleFinder;
  private Configuration conf;

  public PythonViolationsSensor(RuleFinder ruleFinder, Configuration conf) {
    this.ruleFinder = ruleFinder;
    this.conf = conf;
  }

  protected void analyzeFile(InputFile inputFile, ProjectFileSystem projectFileSystem, SensorContext sensorContext) throws IOException {
    Resource pyfile = PythonFile.fromIOFile(inputFile.getFile(), projectFileSystem.getSourceDirs());
    String pylintConfigPath = conf.getString(PythonPlugin.PYLINT_CONFIG_KEY, null);
    String pylintPath = conf.getString(PythonPlugin.PYLINT_KEY, null);
    List<Issue> issues = new PythonViolationsAnalyzer(pylintPath, pylintConfigPath).analyze(inputFile.getFile().getPath());
    for (Issue issue : issues) {
      Rule rule = ruleFinder.findByKey(PythonRuleRepository.REPOSITORY_KEY, issue.ruleId);
      Violation violation = Violation.create(rule, pyfile);
      violation.setLineId(issue.line);
      violation.setMessage(issue.descr);
      sensorContext.saveViolation(violation);
    }
  }
}
