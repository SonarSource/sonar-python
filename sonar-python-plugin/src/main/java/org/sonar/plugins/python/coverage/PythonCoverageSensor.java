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
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import javax.xml.stream.XMLStreamException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.coverage.NewCoverage;
import org.sonar.api.config.Settings;
import org.sonar.plugins.python.EmptyReportException;
import org.sonar.plugins.python.Python;

import static org.sonar.plugins.python.PythonReportSensor.getReports;

public class PythonCoverageSensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonCoverageSensor.class);

  public static final String REPORT_PATH_KEY = "sonar.python.coverage.reportPath";
  public static final String IT_REPORT_PATH_KEY = "sonar.python.coverage.itReportPath";
  public static final String OVERALL_REPORT_PATH_KEY = "sonar.python.coverage.overallReportPath";
  public static final String DEFAULT_REPORT_PATH = "coverage-reports/coverage-*.xml";
  public static final String IT_DEFAULT_REPORT_PATH = "coverage-reports/it-coverage-*.xml";
  public static final String OVERALL_DEFAULT_REPORT_PATH = "coverage-reports/overall-coverage-*.xml";
  public static final String FORCE_ZERO_COVERAGE_KEY = "sonar.python.coverage.forceZeroCoverage";

  public void execute(SensorContext context, Map<InputFile, Set<Integer>> linesOfCode) {
    String baseDir = context.fileSystem().baseDir().getPath();
    Settings settings = context.settings();

    HashSet<InputFile> filesCovered = new HashSet<>();
    List<File> reports = new ArrayList<>();
    reports.addAll(getReports(settings, baseDir, REPORT_PATH_KEY, DEFAULT_REPORT_PATH));
    reports.addAll(getReports(settings, baseDir, IT_REPORT_PATH_KEY, IT_DEFAULT_REPORT_PATH));
    reports.addAll(getReports(settings, baseDir, OVERALL_REPORT_PATH_KEY, OVERALL_DEFAULT_REPORT_PATH));
    if (!reports.isEmpty()) {
      LOG.info("Python test coverage");
      for (File report : reports) {
        Map<InputFile, NewCoverage> coverageMeasures = parseReport(report, context);
        saveMeasures(coverageMeasures, filesCovered);
      }
    }
    if (settings.getBoolean(FORCE_ZERO_COVERAGE_KEY)) {
      LOG.debug("Zeroing coverage information for untouched files");
      zeroMeasuresWithoutReports(context, filesCovered, linesOfCode);
    }
  }

  private static void zeroMeasuresWithoutReports(SensorContext context, HashSet<InputFile> filesCovered, Map<InputFile, Set<Integer>> linesOfCode) {
    FileSystem fileSystem = context.fileSystem();
    FilePredicates p = fileSystem.predicates();
    Iterable<InputFile> inputFiles = fileSystem.inputFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    for (InputFile inputFile : inputFiles) {
      if (!filesCovered.contains(inputFile)) {
        saveZeroValueForResource(inputFile, context, linesOfCode.get(inputFile));
      }
    }
  }

  private static void saveZeroValueForResource(InputFile inputFile, SensorContext context, @Nullable Set<Integer> linesOfCode) {
    if (linesOfCode != null) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Zeroing coverage measures for file '{}'", inputFile.relativePath());
      }
      NewCoverage newCoverage = context.newCoverage()
        .onFile(inputFile);
      linesOfCode.forEach((Integer line) -> newCoverage.lineHits(line, 0));
      newCoverage.save();
    }
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
