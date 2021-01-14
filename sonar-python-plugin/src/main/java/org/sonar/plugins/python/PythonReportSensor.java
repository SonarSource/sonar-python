/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.File;
import java.util.List;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.config.Configuration;
import org.sonar.api.utils.WildcardPattern;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;

public abstract class PythonReportSensor implements Sensor {

  private static final Logger LOG = Loggers.get(PythonReportSensor.class);

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
      List<File> reports = getReports(conf, context.fileSystem().baseDir().getPath(), reportPathPropertyKey, reportPath);
      processReports(context, reports);
    } catch (Exception e) {
      LOG.warn("Cannot read report '{}', the following exception occurred: {}", reportPath, e.getMessage());
      analysisWarnings.addWarning(String.format("An error occurred while trying to import %s report(s): '%s'", reportType, reportPath));
    }
  }

  public static List<File> getReports(Configuration conf, String baseDirPath, String reportPathPropertyKey, String reportPath) {
    LOG.debug("Using pattern '{}' to find reports", reportPath);

    DirectoryScanner scanner = new DirectoryScanner(new File(baseDirPath), WildcardPattern.create(reportPath));
    List<File> includedFiles = scanner.getIncludedFiles();

    if (includedFiles.isEmpty()) {
      if (conf.hasKey(reportPathPropertyKey)) {
        // try absolute path
        File file = new File(reportPath);
        if (!file.exists()) {
          LOG.warn("No report was found for {} using pattern {}", reportPathPropertyKey, reportPath);
        } else {
          includedFiles.add(file);
        }
      } else {
        LOG.debug("No report was found for {} using default pattern {}", reportPathPropertyKey, reportPath);
      }
    }
    return includedFiles;
  }

  protected void processReports(SensorContext context, List<File> reports) throws javax.xml.stream.XMLStreamException {
  }

  protected abstract String reportPathKey();

  protected abstract String defaultReportPath();

}
