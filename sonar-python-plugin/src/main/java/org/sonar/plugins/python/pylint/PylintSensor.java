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
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.ActiveRule;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.config.Configuration;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.Python;

public class PylintSensor implements Sensor {

  private static final Logger LOG = Loggers.get(PylintSensor.class);

  private final PylintConfiguration conf;
  private final Configuration settings;
  private PylintIssuesAnalyzer analyzer;

  public PylintSensor(PylintConfiguration conf, Configuration settings) {
    this.conf = conf;
    this.settings = settings;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .name("PylintSensor")
      .onlyOnLanguage(Python.KEY)
      .onlyOnFileType(InputFile.Type.MAIN)
      .createIssuesForRuleRepository(PylintRuleRepository.REPOSITORY_KEY);
  }

  boolean shouldExecute() {
    return !settings.get(PylintImportSensor.REPORT_PATH_KEY).isPresent();
  }

  @Override
  public void execute(SensorContext sensorContext) {
    File workDir = new File(sensorContext.fileSystem().workDir(), "/pylint/");

    if (!shouldExecute() || !prepareWorkDir(workDir) || !initializeAnalyzer(sensorContext)) {
      return;
    }
    
    int i = 0;
    FileSystem fileSystem = sensorContext.fileSystem();
    FilePredicates p = fileSystem.predicates();
    Iterable<InputFile> files = fileSystem.inputFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    for (InputFile file : files) {
      try {
        File out = new File(workDir, i + ".out");
        analyzeFile(sensorContext, file, out);
        i++;
      } catch (Exception e) {
        LOG.warn("Cannot analyse file '{}', the following exception occurred:", file.toString(), e);
      }
    }
  }

  private boolean initializeAnalyzer(SensorContext context) {
    try {
      String pylintConfigPath = conf.getPylintConfigPath(context.fileSystem());
      String pylintPath = conf.getPylintPath();
      analyzer = createAnalyzer(pylintConfigPath, pylintPath);
      return true;
    } catch (Exception e) {
      LOG.warn("Unable to use pylint for analysis. Error:", e);
      return false;
    }
  }

  // Visible for testing
  PylintIssuesAnalyzer createAnalyzer(String pylintConfigPath, String pylintPath) {
    return new PylintIssuesAnalyzer(pylintPath, pylintConfigPath);
  }

  private void analyzeFile(SensorContext context, InputFile file, File out) throws IOException {
    FileSystem fileSystem = context.fileSystem();

    List<Issue> issues = analyzer.analyze(file.absolutePath(), fileSystem.encoding(), out);

    for (Issue pylintIssue : issues) {
      ActiveRule rule = context.activeRules().find(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, pylintIssue.getRuleId()));
      PylintImportSensor.processRule(pylintIssue, file, rule, context);
    }
  }

  private static boolean prepareWorkDir(File dir) {
    try {
      FileUtils.forceMkdir(dir);
      // directory is cleaned, because Sonar 3.0 will not do this for us
      FileUtils.cleanDirectory(dir);
      return true;
    } catch (IOException e) {
      LOG.warn("Cannot create directory '{}'. Error:", dir, e);
      return false;
    }
  }

}
