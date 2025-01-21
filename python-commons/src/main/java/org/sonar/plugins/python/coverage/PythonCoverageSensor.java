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
package org.sonar.plugins.python.coverage;

import java.io.File;
import java.util.Arrays;
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
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.batch.sensor.coverage.NewCoverage;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.EmptyReportException;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonReportSensor;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;

public class PythonCoverageSensor implements Sensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonCoverageSensor.class);

  public static final String REPORT_PATHS_KEY = "sonar.python.coverage.reportPaths";
  public static final String DEFAULT_REPORT_PATH = "coverage-reports/*coverage-*.xml";

  // Deprecated report path key
  public static final String REPORT_PATH_KEY = "sonar.python.coverage.reportPath";

  private final AnalysisWarningsWrapper analysisWarnings;

  public PythonCoverageSensor(AnalysisWarningsWrapper analysisWarnings) {
    this.analysisWarnings = analysisWarnings;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .name("Cobertura Sensor for Python coverage")
      .onlyOnLanguage(Python.KEY);
  }

  @Override
  public void execute(SensorContext context) {
    String baseDir = context.fileSystem().baseDir().getPath();
    Configuration config = context.config();

    warnDeprecatedPropertyUsage(config);

    try {
      HashSet<InputFile> filesCovered = new HashSet<>();
      List<File> reports = getCoverageReports(baseDir, config);
      if (!reports.isEmpty()) {
        LOG.info("Python test coverage");
        for (File report : uniqueAbsolutePaths(reports)) {
          importReport(context, report, filesCovered);
        }
      }
    } catch (Exception e) {
      LOG.warn("Cannot read coverage report, the following exception occurred: '{}'", e.getMessage());
      analysisWarnings.addUnique("An error occurred while trying to import coverage report(s)");
    }
  }

  private void importReport(SensorContext context, File report, HashSet<InputFile> filesCovered) {
    try {
      Map<InputFile, NewCoverage> coverageMeasures = parseReport(report, context);
      saveMeasures(coverageMeasures, filesCovered);
    } catch (Exception e) {
      LOG.warn("Cannot read coverage report '{}', the following exception occurred: '{}'", report, e.getMessage());
      analysisWarnings.addUnique(String.format("An error occurred while trying to import the coverage report: '%s'", report));
    }
  }

  private List<File> getCoverageReports(String baseDir, Configuration config) {
    if (!config.hasKey(REPORT_PATHS_KEY)) {
      return PythonReportSensor.getReports(config, baseDir, REPORT_PATHS_KEY, DEFAULT_REPORT_PATH, analysisWarnings);
    }

    return Arrays.stream(config.getStringArray(REPORT_PATHS_KEY))
      .flatMap(path -> PythonReportSensor.getReports(config, baseDir, REPORT_PATHS_KEY, path, analysisWarnings).stream())
      .toList();
  }

  private void warnDeprecatedPropertyUsage(Configuration config) {
    if (config.hasKey(REPORT_PATH_KEY)) {
      String msg = "Property 'sonar.python.coverage.reportPath' has been removed. Please use 'sonar.python.coverage.reportPaths' instead.";
      LOG.warn(msg);
      analysisWarnings.addUnique(msg);
    }
  }

  private static Set<File> uniqueAbsolutePaths(List<File> reports) {
    return reports.stream()
      .map(File::getAbsoluteFile)
      .collect(Collectors.toCollection(LinkedHashSet::new));
  }

  private Map<InputFile, NewCoverage> parseReport(File report, SensorContext context) {
    Map<InputFile, NewCoverage> coverageMeasures = new HashMap<>();
    try {
      CoberturaParser parser = new CoberturaParser();
      parser.parseReport(report, context, coverageMeasures);
      if (!parser.errors().isEmpty()) {
        String parseErrors = String.format(String.join("%n", parser.errors()));
        analysisWarnings.addUnique(String.format("The following error(s) occurred while trying to import coverage report:%n%s",
          parseErrors));
      }
    } catch (EmptyReportException e) {
      analysisWarnings.addUnique(String.format("The coverage report '%s' has been ignored because it seems to be empty.", report));
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
        LOG.debug("Saving coverage measures for file '{}'", inputFile);
      }
      entry.getValue()
        .save();
    }
  }
}
