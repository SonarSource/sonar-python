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
package org.sonar.plugins.python.coverage;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.Properties;
import org.sonar.api.Property;
import org.sonar.api.PropertyType;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.CoverageMeasuresBuilder;
import org.sonar.api.measures.Measure;
import org.sonar.api.measures.Metric;
import org.sonar.api.measures.PropertiesBuilder;
import org.sonar.api.resources.Project;
import org.sonar.plugins.python.EmptyReportException;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonReportSensor;

import javax.annotation.Nullable;
import javax.xml.stream.XMLStreamException;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.HashSet;

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
        description = "Path to coverage reports for integration tests, relative to project's root. Ant patterns are accepted. " +
            "The reports have to conform to the Cobertura XML format.",
        global = false,
        project = true),
    @Property(
        key = PythonCoverageSensor.OVERALL_REPORT_PATH_KEY,
        defaultValue = PythonCoverageSensor.OVERALL_DEFAULT_REPORT_PATH,
        name = "Path to overall (combined UT+IT) coverage report(s)",
        description = "Path to a report containing overall test coverage data (i.e. test coverage gained by all tests of all kinds), relative to projects root. " +
            "Ant patterns are accepted. The reports have to conform to the Cobertura XML format.",
        global = false,
        project = true),
    @Property(
        key = PythonCoverageSensor.FORCE_ZERO_COVERAGE_KEY,
        type = PropertyType.BOOLEAN,
        defaultValue = "false",
        name = "Assign zero line coverage to source files without coverage report(s)",
        description = "If 'True', assign zero line coverage to source files without coverage report(s), which results in a more realistic overall Technical Debt value.",
        global = false,
        project = true)
})
public class PythonCoverageSensor extends PythonReportSensor {

  private static final Logger LOG = LoggerFactory.getLogger(PythonCoverageSensor.class);

  private enum CoverageType {
    UT_COVERAGE, IT_COVERAGE, OVERALL_COVERAGE
  }

  public static final String REPORT_PATH_KEY = "sonar.python.coverage.reportPath";
  public static final String IT_REPORT_PATH_KEY = "sonar.python.coverage.itReportPath";
  public static final String OVERALL_REPORT_PATH_KEY = "sonar.python.coverage.overallReportPath";
  public static final String DEFAULT_REPORT_PATH = "coverage-reports/coverage-*.xml";
  public static final String IT_DEFAULT_REPORT_PATH = "coverage-reports/it-coverage-*.xml";
  public static final String OVERALL_DEFAULT_REPORT_PATH = "coverage-reports/overall-coverage-*.xml";
  public static final String FORCE_ZERO_COVERAGE_KEY = "sonar.python.coverage.forceZeroCoverage";

  private CoberturaParser parser = new CoberturaParser();

  public PythonCoverageSensor(Settings conf, FileSystem fileSystem) {
    super(conf, fileSystem);
  }

  @Override
  public void analyse(Project project, SensorContext context) {
    String baseDir = fileSystem.baseDir().getPath();

    List<File> reports = getReports(conf, baseDir, REPORT_PATH_KEY, DEFAULT_REPORT_PATH);
    LOG.debug("Parsing coverage reports");
    Map<String, CoverageMeasuresBuilder> coverageMeasures = parseReports(reports);
    HashSet filesCoveredByUT = new HashSet();
    saveMeasures(context, coverageMeasures, filesCoveredByUT, CoverageType.UT_COVERAGE);

    LOG.debug("Parsing integration test coverage reports");
    List<File> itReports = getReports(conf, baseDir, IT_REPORT_PATH_KEY, IT_DEFAULT_REPORT_PATH);
    Map<String, CoverageMeasuresBuilder> itCoverageMeasures = parseReports(itReports);
    HashSet filesCoveredByIT = new HashSet();
    saveMeasures(context, itCoverageMeasures, filesCoveredByIT, CoverageType.IT_COVERAGE);

    LOG.debug("Parsing overall test coverage reports");
    List<File> overallReports = getReports(conf, baseDir, OVERALL_REPORT_PATH_KEY, OVERALL_DEFAULT_REPORT_PATH);
    Map<String, CoverageMeasuresBuilder> overallCoverageMeasures = parseReports(overallReports);
    HashSet filesCoveredOverall = new HashSet();
    saveMeasures(context, overallCoverageMeasures, filesCoveredOverall, CoverageType.OVERALL_COVERAGE);

    if (conf.getBoolean(FORCE_ZERO_COVERAGE_KEY)) {
      LOG.debug("Zeroing coverage information for untouched files");

      zeroMeasuresWithoutReports(context, filesCoveredByUT, filesCoveredByIT, filesCoveredOverall);
    }
  }

  private void zeroMeasuresWithoutReports(SensorContext context,
                                          HashSet filesCoveredByUT,
                                          HashSet filesCoveredByIT,
                                          HashSet filesCoveredOverall
    ) {
    FilePredicates p = fileSystem.predicates();
    Iterable<InputFile> inputFiles = fileSystem.inputFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
    for (InputFile inputFile : inputFiles) {
      String filePath = inputFile.relativePath();

      if (!filesCoveredByUT.contains(filePath)) {
        saveZeroValueForResource(inputFile, filePath, context, CoverageType.UT_COVERAGE);
      }

      if (!filesCoveredByIT.contains(filePath)) {
        saveZeroValueForResource(inputFile, filePath, context, CoverageType.IT_COVERAGE);
      }

      if (!filesCoveredOverall.contains(filePath)) {
        saveZeroValueForResource(inputFile, filePath, context, CoverageType.OVERALL_COVERAGE);
      }
    }
  }

  private void saveZeroValueForResource(InputFile inputFile,
                                        String filePath, SensorContext context,
                                        CoverageType ctype) {
    Measure ncloc = context.getMeasure(context.getResource(inputFile), CoreMetrics.NCLOC);
    if (ncloc != null && ncloc.getValue() > 0) {
      String coverageKind = "unit test";
      Metric hitsDataMetric = CoreMetrics.COVERAGE_LINE_HITS_DATA;
      Metric linesToCoverMetric = CoreMetrics.LINES_TO_COVER;
      Metric uncoveredLinesMetric = CoreMetrics.UNCOVERED_LINES;

      switch (ctype) {
        case IT_COVERAGE:
          coverageKind = "integration test";
          hitsDataMetric = CoreMetrics.IT_COVERAGE_LINE_HITS_DATA;
          linesToCoverMetric = CoreMetrics.IT_LINES_TO_COVER;
          uncoveredLinesMetric = CoreMetrics.IT_UNCOVERED_LINES;
          break;
        case OVERALL_COVERAGE:
          coverageKind = "overall";
          hitsDataMetric = CoreMetrics.OVERALL_COVERAGE_LINE_HITS_DATA;
          linesToCoverMetric = CoreMetrics.OVERALL_LINES_TO_COVER;
          uncoveredLinesMetric = CoreMetrics.OVERALL_UNCOVERED_LINES;
          break;
        default:
          break;
      }

      LOG.debug("Zeroing {} coverage measures for file '{}'", coverageKind, filePath);

      PropertiesBuilder<Integer, Integer> lineHitsData = new PropertiesBuilder<>(hitsDataMetric);
      for (int i = 1; i <= inputFile.lines(); ++i) {
        lineHitsData.add(i, 0);
      }
      context.saveMeasure(inputFile, lineHitsData.build());
      context.saveMeasure(inputFile, linesToCoverMetric, ncloc.getValue());
      context.saveMeasure(inputFile, uncoveredLinesMetric, ncloc.getValue());
    }
  }


  private Map<String, CoverageMeasuresBuilder> parseReports(List<File> reports) {
    Map<String, CoverageMeasuresBuilder> coverageMeasures = new HashMap<String, CoverageMeasuresBuilder>();
    for (File report : reports) {
      try {
        parser.parseReport(report, coverageMeasures);
      } catch (EmptyReportException e) {
        LOG.warn("The report '{}' seems to be empty, ignoring. '{}'", report, e);
      } catch (XMLStreamException e) {
        throw new IllegalStateException("Error parsing the report '" + report + "'", e);
      }
    }
    return coverageMeasures;
  }

  private void saveMeasures(SensorContext context,
                            Map<String, CoverageMeasuresBuilder> coverageMeasures,
                            HashSet coveredFiles,
                            CoverageType coverageType) {
    for (Map.Entry<String, CoverageMeasuresBuilder> entry : coverageMeasures.entrySet()) {
      String filePath = entry.getKey();
      InputFile pythonFile = fileSystem.inputFile(fileSystem.predicates().hasPath(filePath));
      if (pythonFile != null) {
        coveredFiles.add(pythonFile.relativePath());

        LOG.debug("Saving coverage measures for file '{}'", filePath);
        for (Measure measure : entry.getValue().createMeasures()) {
          switch (coverageType) {
            case IT_COVERAGE:
              measure = convertToITMeasure(measure);
              break;
            case OVERALL_COVERAGE:
              measure = convertForOverall(measure);
              break;
            default:
              break;
          }
          context.saveMeasure(pythonFile, measure);
        }
      } else {
        LOG.debug("Cannot find the file '{}', ignoring coverage measures", filePath);
      }
    }
  }

  Measure convertToITMeasure(Measure measure) {
    Measure itMeasure = null;
    Metric metric = measure.getMetric();
    Double value = measure.getValue();
    String data = measure.getData();
    if (CoreMetrics.LINES_TO_COVER.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_LINES_TO_COVER, value);
    } else if (CoreMetrics.UNCOVERED_LINES.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_UNCOVERED_LINES, value);
    } else if (CoreMetrics.COVERAGE_LINE_HITS_DATA.equals(metric)) {
      checkDataIsNotNull(data);
      itMeasure = new Measure(CoreMetrics.IT_COVERAGE_LINE_HITS_DATA, data);
    } else if (CoreMetrics.CONDITIONS_TO_COVER.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_CONDITIONS_TO_COVER, value);
    } else if (CoreMetrics.UNCOVERED_CONDITIONS.equals(metric)) {
      itMeasure = new Measure(CoreMetrics.IT_UNCOVERED_CONDITIONS, value);
    } else if (CoreMetrics.COVERED_CONDITIONS_BY_LINE.equals(metric)) {
      checkDataIsNotNull(data);
      itMeasure = new Measure(CoreMetrics.IT_COVERED_CONDITIONS_BY_LINE, data);
    } else if (CoreMetrics.CONDITIONS_BY_LINE.equals(metric)) {
      checkDataIsNotNull(data);
      itMeasure = new Measure(CoreMetrics.IT_CONDITIONS_BY_LINE, data);
    }
    return itMeasure;
  }

  private Measure convertForOverall(Measure measure) {
    Measure overallMeasure = null;
    String data = measure.getData();
    if (CoreMetrics.LINES_TO_COVER.equals(measure.getMetric())) {
      overallMeasure = new Measure(CoreMetrics.OVERALL_LINES_TO_COVER, measure.getValue());
    } else if (CoreMetrics.UNCOVERED_LINES.equals(measure.getMetric())) {
      overallMeasure = new Measure(CoreMetrics.OVERALL_UNCOVERED_LINES, measure.getValue());
    } else if (CoreMetrics.COVERAGE_LINE_HITS_DATA.equals(measure.getMetric())) {
      checkDataIsNotNull(data);
      overallMeasure = new Measure(CoreMetrics.OVERALL_COVERAGE_LINE_HITS_DATA, data);
    } else if (CoreMetrics.CONDITIONS_TO_COVER.equals(measure.getMetric())) {
      overallMeasure = new Measure(CoreMetrics.OVERALL_CONDITIONS_TO_COVER, measure.getValue());
    } else if (CoreMetrics.UNCOVERED_CONDITIONS.equals(measure.getMetric())) {
      overallMeasure = new Measure(CoreMetrics.OVERALL_UNCOVERED_CONDITIONS, measure.getValue());
    } else if (CoreMetrics.COVERED_CONDITIONS_BY_LINE.equals(measure.getMetric())) {
      checkDataIsNotNull(data);
      overallMeasure = new Measure(CoreMetrics.OVERALL_COVERED_CONDITIONS_BY_LINE, data);
    } else if (CoreMetrics.CONDITIONS_BY_LINE.equals(measure.getMetric())) {
      checkDataIsNotNull(data);
      overallMeasure = new Measure(CoreMetrics.OVERALL_CONDITIONS_BY_LINE, data);
    }

    return overallMeasure;
  }

  private void checkDataIsNotNull(@Nullable String data) {
    if (data == null) {
      throw new IllegalStateException("Measure data is null but it shouldn't be");
    }
  }
}
