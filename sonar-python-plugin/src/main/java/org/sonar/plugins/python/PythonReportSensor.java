/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Settings;
import org.sonar.api.resources.Project;
import org.sonar.api.utils.WildcardPattern;

import java.io.File;
import java.util.List;

public abstract class PythonReportSensor implements Sensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonReportSensor.class);

  protected Settings conf = null;
  protected FileSystem fileSystem;

  public PythonReportSensor(Settings conf, FileSystem fileSystem) {
    this.conf = conf;
    this.fileSystem = fileSystem;
  }

  @Override
  public boolean shouldExecuteOnProject(Project project) {
    FilePredicates p = fileSystem.predicates();
    return fileSystem.hasFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
  }

  @Override
  public void analyse(Project project, SensorContext context) {
    try {
      List<File> reports = getReports(conf, fileSystem.baseDir().getPath(), reportPathKey(), defaultReportPath());
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

  @Override
  public String toString() {
    return getClass().getSimpleName();
  }

  protected List<File> getReports(Settings conf, String baseDirPath, String reportPathPropertyKey, String defaultReportPath) {
    String reportPath = conf.getString(reportPathPropertyKey);
    if (reportPath == null) {
      reportPath = defaultReportPath;
    }

    LOG.debug("Using pattern '{}' to find reports", reportPath);

    DirectoryScanner scanner = new DirectoryScanner(new File(baseDirPath), WildcardPattern.create(reportPath));
    return scanner.getIncludedFiles();
  }

  protected void processReports(SensorContext context, List<File> reports) throws javax.xml.stream.XMLStreamException {
  }

  protected String reportPathKey() {
    return "";
  }

  protected String defaultReportPath() {
    return "";
  }

}
