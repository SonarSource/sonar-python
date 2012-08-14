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
package org.sonar.plugins.python;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.tools.ant.DirectoryScanner;
import org.apache.commons.configuration.Configuration;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.resources.Project;
import org.sonar.api.utils.SonarException;
import org.sonar.plugins.python.Python;

/**
 * {@inheritDoc}
 */
public abstract class PythonReportSensor implements Sensor {
  private Configuration conf = null;

  /**
   * {@inheritDoc}
   */
  public PythonReportSensor(Configuration conf) {
    this.conf = conf;
  }
  
  /**
   * {@inheritDoc}
   */
  public boolean shouldExecuteOnProject(Project project) {
    return Python.KEY.equals(project.getLanguageKey());
  }

  /**
   * {@inheritDoc}
   */
  public void analyse(Project project, SensorContext context) {
    try {
      List<File> reports = getReports(conf, project.getFileSystem().getBasedir().getPath(),
                                      reportPathKey(), defaultReportPath());
      for (File report : reports) {
        PythonPlugin.LOG.info("Processing report '{}'", report);
        processReport(project, context, report);
      }
      
      if (reports.isEmpty()) {
        handleNoReportsCase(context);
      }
    } catch (javax.xml.stream.XMLStreamException e) {
      String msg = new StringBuilder()
        .append("Cannot feed the data into sonar, details: '")
        .append(e)
        .append("'")
        .toString();
      throw new SonarException(msg, e);
    }
  }

  @Override
  public String toString() {
    return getClass().getSimpleName();
  }

  protected List<File> getReports(Configuration conf,
                                  String baseDirPath,
                                  String reportPathPropertyKey,
                                  String defaultReportPath) {
    String reportPath = conf.getString(reportPathPropertyKey, null);
    if(reportPath == null){
      reportPath = defaultReportPath;
    }
    
    PythonPlugin.LOG.debug("Using pattern '{}' to find reports", reportPath);

    DirectoryScanner scanner = new DirectoryScanner();
    String[] includes = { reportPath };
    scanner.setIncludes(includes);
    scanner.setBasedir(new File(baseDirPath));
    scanner.scan();
    String[] relPaths = scanner.getIncludedFiles();

    List<File> reports = new ArrayList<File>();
    for (String relPath : relPaths) {
      reports.add(new File(baseDirPath, relPath));
    }
    
    return reports;
  }

  protected void processReport(Project project, SensorContext context, File report)
    throws
    javax.xml.stream.XMLStreamException
  {}

  protected void handleNoReportsCase(SensorContext context) {}
  
  protected String reportPathKey() { return ""; };
  
  protected String defaultReportPath() { return ""; };
}
