/*
 * Sonar Python Plugin
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

import org.apache.commons.lang.StringUtils;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.File;
import org.sonar.api.resources.InputFile;
import org.sonar.api.resources.Project;
import org.sonar.api.rules.Rule;
import org.sonar.api.rules.RuleFinder;
import org.sonar.api.rules.Violation;
import org.sonar.api.utils.SonarException;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonPlugin;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class PylintSensor implements Sensor {
  private static final String PYTHONPATH_ENVVAR = "PYTHONPATH";

  private RuleFinder ruleFinder;
  private RulesProfile profile;
  private PylintConfiguration conf;
  private String[] environment;

  public PylintSensor(RuleFinder ruleFinder, Project project, PylintConfiguration conf, RulesProfile profile) {
    this.ruleFinder = ruleFinder;
    this.conf = conf;
    this.profile = profile;
    this.environment = getEnvironment(project);
  }

  public boolean shouldExecuteOnProject(Project project) {
    return Python.KEY.equals(project.getLanguageKey())
      && !profile.getActiveRulesByRepository(PylintRuleRepository.REPOSITORY_KEY).isEmpty();
  }

  public void analyse(Project project, SensorContext sensorContext) {
    for (InputFile inputFile : project.getFileSystem().mainFiles(Python.KEY)) {
      try {
        analyzeFile(inputFile, project, sensorContext);
      } catch (Exception e) {
        String msg = new StringBuilder()
            .append("Cannot analyse the file '")
            .append(inputFile.getFile().getAbsolutePath())
            .append("', details: '")
            .append(e)
            .append("'")
            .toString();
        throw new SonarException(msg, e);
      }
    }
  }

  protected void analyzeFile(InputFile inputFile, Project project, SensorContext sensorContext) throws IOException {
    File pyfile = File.fromIOFile(inputFile.getFile(), project);

    String pylintConfigPath = conf.getPylintConfigPath(project);
    String pylintPath = conf.getPylintPath();

    PylintViolationsAnalyzer analyzer = new PylintViolationsAnalyzer(pylintPath, pylintConfigPath);
    List<Issue> issues = analyzer.analyze(inputFile.getFile().getPath(), environment);
    for (Issue issue : issues) {
      Rule rule = ruleFinder.findByKey(PylintRuleRepository.REPOSITORY_KEY, issue.ruleId);
      if (rule != null) {
        if (rule.isEnabled()) {
          Violation violation = Violation.create(rule, pyfile);
          violation.setLineId(issue.line);
          violation.setMessage(issue.descr);
          sensorContext.saveViolation(violation);
        } else {
          PythonPlugin.LOG.debug("Pylint rule '{}' is disabled in Sonar",  issue.ruleId);
        }
      } else {
        PythonPlugin.LOG.warn("Pylint rule '{}' is unknown in Sonar",  issue.ruleId);
      }
    }
  }

  protected static final String[] getEnvironment(Project project){
    String[] environ = null;
    String pythonPathProp = (String) project.getProperty(PylintConfiguration.PYTHON_PATH_KEY);
    if (pythonPathProp != null){
      java.io.File projectRoot = project.getFileSystem().getBasedir();
      String[] parsedPaths = StringUtils.split(pythonPathProp, ",");
      List<String> absPaths = toAbsPaths(parsedPaths, projectRoot);
      String delimiter = System.getProperty("path.separator");
      String pythonPath = StringUtils.join(absPaths, delimiter);

      environ = new String[1];
      environ[0] = PYTHONPATH_ENVVAR + "=" + pythonPath;
    }
    return environ;
  }


  private static List<String> toAbsPaths(String[] pathStrings, java.io.File baseDir){
    List<String> result = new LinkedList<String>();
    for(String pathStr: pathStrings){
      pathStr = StringUtils.trim(pathStr);
      result.add(new java.io.File(baseDir, pathStr).getAbsolutePath());
    }
    return result;
  }

}
