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
package org.sonar.plugins.python.coverage;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.xml.stream.XMLStreamException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.coverage.NewCoverage;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.EmptyReportException;

import static org.sonar.plugins.python.PythonReportSensor.getReports;

public class PythonCoverageSensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonCoverageSensor.class);

  public static final String REPORT_PATH_KEY = "sonar.python.coverage.reportPath";
  public static final String DEFAULT_REPORT_PATH = "coverage-reports/*coverage-*.xml";
  // Deprecated report path keys
  public static final String IT_REPORT_PATH_KEY = "sonar.python.coverage.itReportPath";
  public static final String IT_DEFAULT_REPORT_PATH = "";
  public static final String OVERALL_REPORT_PATH_KEY = "sonar.python.coverage.overallReportPath";
  public static final String OVERALL_DEFAULT_REPORT_PATH = "";

  public void execute(SensorContext context) {
    String baseDir = context.fileSystem().baseDir().getPath();
    Configuration config = context.config();

    logDeprecatedPropertyUsage(config, IT_REPORT_PATH_KEY, REPORT_PATH_KEY);
    logDeprecatedPropertyUsage(config, OVERALL_REPORT_PATH_KEY, REPORT_PATH_KEY);

    HashSet<InputFile> filesCovered = new HashSet<>();
    List<File> reports = new ArrayList<>();
    reports.addAll(getReports(config, baseDir, REPORT_PATH_KEY, DEFAULT_REPORT_PATH));
    reports.addAll(getReports(config, baseDir, IT_REPORT_PATH_KEY, IT_DEFAULT_REPORT_PATH));
    reports.addAll(getReports(config, baseDir, OVERALL_REPORT_PATH_KEY, OVERALL_DEFAULT_REPORT_PATH));
    if (!reports.isEmpty()) {
      LOG.info("Python test coverage");
      for (File report : uniqueAbsolutePaths(reports)) {
        Map<InputFile, NewCoverage> coverageMeasures = parseReport(report, context);
        saveMeasures(coverageMeasures, filesCovered);
      }
    }
  }

  private static void logDeprecatedPropertyUsage(Configuration config, String deprecatedKey, String replacementKey) {
    if (!config.get(deprecatedKey).orElse("").isEmpty()) {
      LOG.warn("Property '{}' is deprecated. Please use '{}' instead.", deprecatedKey, replacementKey);
    }
  }

  private static Set<File> uniqueAbsolutePaths(List<File> reports) {
    return reports.stream()
      .map(File::getAbsoluteFile)
      .collect(Collectors.toCollection(LinkedHashSet::new));
  }

  private static Map<InputFile, NewCoverage> parseReport(File report, SensorContext context) {
    Map<InputFile, NewCoverage> coverageMeasures = new HashMap<>();
    try {
      CoberturaParser parser = new CoberturaParser();
      parser.parseReport(report, context, coverageMeasures);
    } catch (EmptyReportException e) {
      LOG.warn("The report '{}' seems to be empty, ignoring. '{}'", report, e);
    } catch (XMLStreamException e) {
      throw new IllegalStateException("Error parsing the report '" + report + "'", e);
    }
    return coverageMeasures;
  }

  private static void saveMeasures(Map<InputFile, NewCoverage> coverageMeasures, HashSet<InputFile> coveredFiles) {
    for (Map.Entry<InputFile, NewCoverage> entry : coverageMeasures.entrySet()) {
      InputFile inputFile = entry.getKey();
      coveredFiles.add(inputFile);
      if (LOG.isDebugEnabled()) {
        LOG.debug("Saving coverage measures for file '{}'", inputFile.relativePath());
      }
      entry.getValue()
        .save();
    }
  }
}
