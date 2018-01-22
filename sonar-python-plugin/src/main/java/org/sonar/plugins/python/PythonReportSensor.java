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
package org.sonar.plugins.python;

import java.io.File;
import java.util.List;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.config.Configuration;
import org.sonar.api.utils.WildcardPattern;

public abstract class PythonReportSensor implements Sensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonReportSensor.class);

  protected Configuration conf;

  public PythonReportSensor(Configuration conf) {
    this.conf = conf;
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
    try {
      List<File> reports = getReports(conf, context.fileSystem().baseDir().getPath(), reportPathKey(), defaultReportPath());
      processReports(context, reports);
    } catch (javax.xml.stream.XMLStreamException e) {
      String msg = new StringBuilder()
          .append("Cannot feed the data into sonar, details: '")
          .append(e)
          .append("'")
          .toString();
      throw new IllegalStateException(msg, e);
    }
  }

  public static List<File> getReports(Configuration conf, String baseDirPath, String reportPathPropertyKey, String defaultReportPath) {
    String reportPath = conf.get(reportPathPropertyKey).orElse(defaultReportPath);
    boolean propertyIsProvided = !Objects.equals(reportPath, defaultReportPath);

    LOG.debug("Using pattern '{}' to find reports", reportPath);

    DirectoryScanner scanner = new DirectoryScanner(new File(baseDirPath), WildcardPattern.create(reportPath));
    List<File> includedFiles = scanner.getIncludedFiles();

    if (includedFiles.isEmpty()) {
      if (propertyIsProvided) {
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
