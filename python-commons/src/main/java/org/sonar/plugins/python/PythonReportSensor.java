/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import java.io.File;
import java.util.List;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.config.Configuration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;
import org.sonarsource.analyzer.commons.FileProvider;

public abstract class PythonReportSensor implements Sensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonReportSensor.class);

  protected final Configuration conf;
  private final AnalysisWarningsWrapper analysisWarnings;
  private final String reportType;

  public PythonReportSensor(Configuration conf, AnalysisWarningsWrapper analysisWarnings, String reportType) {
    this.conf = conf;
    this.analysisWarnings = analysisWarnings;
    this.reportType = reportType;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .name(getClass().getSimpleName())
      .onlyOnLanguage(Python.KEY)
      .onlyOnFileType(InputFile.Type.MAIN);
  }

  @Override
  public void execute(SensorContext context) {
    String reportPathPropertyKey = reportPathKey();
    String reportPath = conf.get(reportPathPropertyKey).orElse(defaultReportPath());
    try {
      List<File> reports = getReports(conf, context.fileSystem().baseDir().getPath(), reportPathPropertyKey, reportPath, analysisWarnings);
      processReports(context, reports);
    } catch (Exception e) {
      LOG.warn("Cannot read report '{}', the following exception occurred: {}", reportPath, e.getMessage());
      analysisWarnings.addUnique(String.format("An error occurred while trying to import %s report(s): '%s'", reportType, reportPath));
    }
  }

  public static List<File> getReports(Configuration conf, String baseDirPath, String reportPathPropertyKey, String reportPath, AnalysisWarningsWrapper analysisWarnings) {
    LOG.debug("Using pattern '{}' to find reports", reportPath);

    FileProvider provider = new FileProvider(new File(baseDirPath), reportPath);
    List<File> matchingFiles = provider.getMatchingFiles();

    if (matchingFiles.isEmpty()) {
      if (conf.hasKey(reportPathPropertyKey)) {
        // try absolute path
        File file = new File(reportPath);
        if (!file.exists()) {
          String formattedMessage = String.format("No report was found for %s using pattern %s", reportPathPropertyKey, reportPath);
          LOG.warn(formattedMessage);
          analysisWarnings.addUnique(formattedMessage);
        } else {
          matchingFiles.add(file);
        }
      } else {
        LOG.debug("No report was found for {} using default pattern {}", reportPathPropertyKey, reportPath);
      }
    }
    return matchingFiles;
  }

  protected void processReports(SensorContext context, List<File> reports) throws javax.xml.stream.XMLStreamException {
  }

  protected abstract String reportPathKey();

  protected abstract String defaultReportPath();

}
