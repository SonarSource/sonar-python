/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
import org.sonar.api.config.Settings;
import org.sonar.api.rule.RuleKey;
import org.sonar.plugins.python.Python;

public class PylintSensor implements Sensor {

  private PylintConfiguration conf;
  private Settings settings;

  public PylintSensor(PylintConfiguration conf, Settings settings) {
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
    return settings.getString(PylintImportSensor.REPORT_PATH_KEY) == null;
  }

  @Override
  public void execute(SensorContext sensorContext) {
    if (!shouldExecute()) {
      return;
    }

    FileSystem fileSystem = sensorContext.fileSystem();
    File workDir = new File(fileSystem.workDir(), "/pylint/");
    prepareWorkDir(workDir);
    int i = 0;
    FilePredicates p = fileSystem.predicates();
    Iterable<InputFile> files = fileSystem.inputFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    for (InputFile file : files) {
      try {
        File out = new File(workDir, i + ".out");
        analyzeFile(sensorContext, file, out);
        i++;
      } catch (Exception e) {
        String msg = new StringBuilder()
            .append("Cannot analyse the file '")
            .append(file.absolutePath())
            .append("', details: '")
            .append(e)
            .append("'")
            .toString();
        throw new IllegalStateException(msg, e);
      }
    }
  }

  private void analyzeFile(SensorContext context, InputFile file, File out) throws IOException {
    FileSystem fileSystem = context.fileSystem();

    String pylintConfigPath = conf.getPylintConfigPath(fileSystem);
    String pylintPath = conf.getPylintPath();

    PylintIssuesAnalyzer analyzer = new PylintIssuesAnalyzer(pylintPath, pylintConfigPath);
    List<Issue> issues = analyzer.analyze(file.absolutePath(), fileSystem.encoding(), out);

    for (Issue pylintIssue : issues) {
      ActiveRule rule = context.activeRules().find(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, pylintIssue.getRuleId()));
      PylintImportSensor.processRule(pylintIssue, file, rule, context);
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
