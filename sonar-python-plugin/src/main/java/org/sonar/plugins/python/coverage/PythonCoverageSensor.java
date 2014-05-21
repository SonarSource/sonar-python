/*
 * SonarQube Python Plugin
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
package org.sonar.plugins.python.coverage;

import org.sonar.api.Properties;
import org.sonar.api.Property;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.CoverageMeasuresBuilder;
import org.sonar.api.measures.Measure;
import org.sonar.api.measures.Metric;
import org.sonar.api.resources.Project;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.api.utils.SonarException;
import org.sonar.plugins.python.PythonReportSensor;

import javax.xml.stream.XMLStreamException;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Properties({
  @Property(
    key = PythonCoverageSensor.REPORT_PATH_KEY,
    defaultValue = PythonCoverageSensor.DEFAULT_REPORT_PATH,
    name = "Path to coverage report(s)",
    description = "Path to coverage reports, relative to project's root. Ant patterns are accepted. The reports have to conform to the Cobertura XML format.",
    global = false,
    project = true),
  @Property(
    key = PythonCoverageSensor.IT_REPORT_PATH_KEY,
    defaultValue = PythonCoverageSensor.IT_DEFAULT_REPORT_PATH,
    name = "Path to coverage report(s) for integration tests",
    description = "Path to coverage reports for integration tests, relative to project's root. Ant patterns are accepted. The reports have to conform to the Cobertura XML format.",
    global = false,
    project = true)
})
public class PythonCoverageSensor extends PythonReportSensor {
  public static final String REPORT_PATH_KEY = "sonar.python.coverage.reportPath";
  public static final String IT_REPORT_PATH_KEY = "sonar.python.coverage.itReportPath";
  public static final String DEFAULT_REPORT_PATH = "coverage-reports/coverage-*.xml";
  public static final String IT_DEFAULT_REPORT_PATH = "coverage-reports/it-coverage-*.xml";

  private CoberturaParser parser = new CoberturaParser();

  public PythonCoverageSensor(Settings conf, ModuleFileSystem fileSystem) {
    super(conf, fileSystem);
  }

  @Override
  public void analyse(Project project, SensorContext context) {
    List<File> reports = getReports(conf, fileSystem.baseDir().getPath(), REPORT_PATH_KEY, DEFAULT_REPORT_PATH);
    LOG.debug("Parsing coverage reports");
    Map<String, CoverageMeasuresBuilder> coverageMeasures = parseReports(reports);
    saveMeasures(project, context, coverageMeasures, false);

    LOG.debug("Parsing integration test coverage reports");
    List<File> itReports = getReports(conf, fileSystem.baseDir().getPath(), IT_REPORT_PATH_KEY, IT_DEFAULT_REPORT_PATH);
    coverageMeasures = parseReports(itReports);
    saveMeasures(project, context, coverageMeasures, true);
  }

  private Map<String, CoverageMeasuresBuilder> parseReports(List<File> reports) {
    Map<String, CoverageMeasuresBuilder>  coverageMeasures = new HashMap<String, CoverageMeasuresBuilder>();
    for (File report : reports) {
      try{
        parser.parseReport(report, coverageMeasures);
      } catch (XMLStreamException e) {
        throw new SonarException("Error parsing the report '" + report + "'", e);
      }
    }
    return coverageMeasures;
  }

  private void saveMeasures(Project project,
                            SensorContext context,
                            Map<String, CoverageMeasuresBuilder> coverageMeasures,
                            boolean itTest) {
    FileResolver fileResolver = new FileResolver(project, fileSystem);
    for(Map.Entry<String, CoverageMeasuresBuilder> entry: coverageMeasures.entrySet()) {
      String filePath = entry.getKey();
      org.sonar.api.resources.File pythonfile = fileResolver.getFile(filePath);
      if (fileExist(context, pythonfile)) {
        LOG.debug("Saving coverage measures for file '{}'", filePath);
        for (Measure measure : entry.getValue().createMeasures()) {
          measure = itTest ? convertToItMeasure(measure) : measure;
          context.saveMeasure(pythonfile, measure);
        }
      } else {
        LOG.debug("Cannot find the file '{}', ignoring coverage measures", filePath);
      }
    }
  }

  Measure convertToItMeasure(Measure measure){
    Measure itMeasure = null;
    Metric metric = measure.getMetric();
    Double value = measure.getValue();
    if (CoreMetrics.LINES_TO_COVER.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_LINES_TO_COVER, value);
    } else if (CoreMetrics.UNCOVERED_LINES.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_UNCOVERED_LINES, value);
    } else if (CoreMetrics.COVERAGE_LINE_HITS_DATA.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_COVERAGE_LINE_HITS_DATA, measure.getData());
    } else if (CoreMetrics.CONDITIONS_TO_COVER.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_CONDITIONS_TO_COVER, value);
    } else if (CoreMetrics.UNCOVERED_CONDITIONS.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_UNCOVERED_CONDITIONS, value);
    } else if (CoreMetrics.COVERED_CONDITIONS_BY_LINE.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_COVERED_CONDITIONS_BY_LINE, measure.getData());
    } else if (CoreMetrics.CONDITIONS_BY_LINE.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_CONDITIONS_BY_LINE, measure.getData());
    }
    return itMeasure;
  }

  private boolean fileExist(SensorContext context, org.sonar.api.resources.File file) {
    return context.getResource(file) != null;
  }

}
