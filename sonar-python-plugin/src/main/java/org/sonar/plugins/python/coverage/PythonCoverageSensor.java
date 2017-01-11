/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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
import org.sonar.api.batch.sensor.coverage.CoverageType;
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

  private CoberturaParser parser = new CoberturaParser();

  public void execute(SensorContext context, Map<InputFile, Set<Integer>> linesOfCode) {
    String baseDir = context.fileSystem().baseDir().getPath();
    Settings settings = context.settings();

    LOG.info("Python unit test coverage");
    List<File> reports = getReports(settings, baseDir, REPORT_PATH_KEY, DEFAULT_REPORT_PATH);
    Map<InputFile, NewCoverage> coverageMeasures = parseReports(reports, context);
    HashSet<InputFile> filesCoveredByUT = new HashSet<>();
    saveMeasures(coverageMeasures, filesCoveredByUT, CoverageType.UNIT);

    LOG.info("Python integration test coverage");
    List<File> itReports = getReports(settings, baseDir, IT_REPORT_PATH_KEY, IT_DEFAULT_REPORT_PATH);
    Map<InputFile, NewCoverage> itCoverageMeasures = parseReports(itReports, context);
    HashSet<InputFile> filesCoveredByIT = new HashSet<>();
    saveMeasures(itCoverageMeasures, filesCoveredByIT, CoverageType.IT);

    LOG.info("Python overall test coverage");
    List<File> overallReports = getReports(settings, baseDir, OVERALL_REPORT_PATH_KEY, OVERALL_DEFAULT_REPORT_PATH);
    Map<InputFile, NewCoverage> overallCoverageMeasures = parseReports(overallReports, context);
    HashSet<InputFile> filesCoveredOverall = new HashSet<>();
    saveMeasures(overallCoverageMeasures, filesCoveredOverall, CoverageType.OVERALL);

    if (settings.getBoolean(FORCE_ZERO_COVERAGE_KEY)) {
      LOG.debug("Zeroing coverage information for untouched files");
      zeroMeasuresWithoutReports(context, filesCoveredByUT, filesCoveredByIT, filesCoveredOverall, linesOfCode);
    }
  }

  private static void zeroMeasuresWithoutReports(
    SensorContext context,
    HashSet<InputFile> filesCoveredByUT,
    HashSet<InputFile> filesCoveredByIT,
    HashSet<InputFile> filesCoveredOverall,
    Map<InputFile, Set<Integer>> linesOfCode
  ) {
    FileSystem fileSystem = context.fileSystem();
    FilePredicates p = fileSystem.predicates();
    Iterable<InputFile> inputFiles = fileSystem.inputFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    for (InputFile inputFile : inputFiles) {
      Set<Integer> linesOfCodeForFile = linesOfCode.get(inputFile);

      if (!filesCoveredByUT.contains(inputFile)) {
        saveZeroValueForResource(inputFile, context, CoverageType.UNIT, linesOfCodeForFile);
      }

      if (!filesCoveredByIT.contains(inputFile)) {
        saveZeroValueForResource(inputFile, context, CoverageType.IT, linesOfCodeForFile);
      }

      if (!filesCoveredOverall.contains(inputFile)) {
        saveZeroValueForResource(inputFile, context, CoverageType.OVERALL, linesOfCodeForFile);
      }
    }
  }

  private static void saveZeroValueForResource(InputFile inputFile, SensorContext context, CoverageType ctype, @Nullable Set<Integer> linesOfCode) {
    if (linesOfCode != null) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Zeroing {} coverage measures for file '{}'", ctype, inputFile.relativePath());
      }

      NewCoverage newCoverage = context.newCoverage()
        .onFile(inputFile)
        .ofType(ctype);
      linesOfCode.forEach((Integer line) -> newCoverage.lineHits(line, 0));
      newCoverage.save();
    }
  }


  private Map<InputFile, NewCoverage> parseReports(List<File> reports, SensorContext context) {
    Map<InputFile, NewCoverage> coverageMeasures = new HashMap<>();
    for (File report : reports) {
      try {
        parser.parseReport(report, context, coverageMeasures);
      } catch (EmptyReportException e) {
        LOG.warn("The report '{}' seems to be empty, ignoring. '{}'", report, e);
      } catch (XMLStreamException e) {
        throw new IllegalStateException("Error parsing the report '" + report + "'", e);
      }
    }
    return coverageMeasures;
  }

  private static void saveMeasures(Map<InputFile, NewCoverage> coverageMeasures, HashSet<InputFile> coveredFiles, CoverageType coverageType) {
    for (Map.Entry<InputFile, NewCoverage> entry : coverageMeasures.entrySet()) {
      InputFile inputFile = entry.getKey();
      coveredFiles.add(inputFile);

      if (LOG.isDebugEnabled()) {
        LOG.debug("Saving coverage measures for file '{}'", inputFile.relativePath());
      }

      entry.getValue()
        .ofType(coverageType)
        .save();

    }
  }
}
