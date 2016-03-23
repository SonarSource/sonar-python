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
package org.sonar.plugins.python.xunit;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.Properties;
import org.sonar.api.Property;
import org.sonar.api.PropertyType;
import org.sonar.api.batch.CoverageExtension;
import org.sonar.api.batch.DependsUpon;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Measure;
import org.sonar.api.utils.ParsingUtils;
import org.sonar.api.utils.StaxParser;
import org.sonar.plugins.python.PythonReportSensor;

import javax.xml.stream.XMLStreamException;
import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Properties({
    @Property(
        key = PythonXUnitSensor.REPORT_PATH_KEY,
        defaultValue = PythonXUnitSensor.DEFAULT_REPORT_PATH,
        name = "Path to xunit report(s)",
        description = "Path to the report of test execution, relative to project's root. Ant patterns are accepted. The reports have to conform to the junitreport XML format.",
        global = false, project = true),

    @Property(
        key = PythonXUnitSensor.SKIP_DETAILS,
        type = PropertyType.BOOLEAN,
        defaultValue = "true",
        name = "Skip the details when importing the Xunit reports",
        description = "If 'true', provides the test execution statistics only on project level, but makes the import procedure more mature",
        global = false, project = true) })
public class PythonXUnitSensor extends PythonReportSensor {
  private static final Logger LOG = LoggerFactory.getLogger(PythonXUnitSensor.class);

  public static final String REPORT_PATH_KEY = "sonar.python.xunit.reportPath";
  public static final String DEFAULT_REPORT_PATH = "xunit-reports/xunit-result-*.xml";
  public static final String SKIP_DETAILS = "sonar.python.xunit.skipDetails";
  private static final double PERCENT_BASE = 100d;

  public PythonXUnitSensor(Settings conf, FileSystem fileSystem) {
    super(conf, fileSystem);
  }

  @DependsUpon
  public Class dependsUponCoverageSensors() {
    return CoverageExtension.class;
  }

  @Override
  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  @Override
  protected String defaultReportPath() {
    return DEFAULT_REPORT_PATH;
  }

  @Override
  protected void processReports(final SensorContext context, List<File> reports) throws XMLStreamException {
    if (conf.getBoolean(SKIP_DETAILS)) {
      simpleMode(context, reports);
    } else {
      detailedMode(context, reports);
    }
  }

  private void simpleMode(final SensorContext context, List<File> reports) throws XMLStreamException {
    TestSuiteParser parserHandler = new TestSuiteParser();
    StaxParser parser = new StaxParser(parserHandler, false);
    for (File report : reports) {
      parser.parse(report);
    }

    double testsCount = 0.0;
    double testsSkipped = 0.0;
    double testsErrors = 0.0;
    double testsFailures = 0.0;
    double testsTime = 0.0;
    for (TestSuite report : parserHandler.getParsedReports()) {
      testsCount += report.getTests() - report.getSkipped();
      testsSkipped += report.getSkipped();
      testsErrors += report.getErrors();
      testsFailures += report.getFailures();
      testsTime += report.getTime();
    }

    if (testsCount > 0) {
      double testsPassed = testsCount - testsErrors - testsFailures;
      double successDensity = testsPassed * PERCENT_BASE / testsCount;
      context.saveMeasure(CoreMetrics.TEST_SUCCESS_DENSITY, ParsingUtils.scaleValue(successDensity));

      context.saveMeasure(CoreMetrics.TESTS, testsCount);
      context.saveMeasure(CoreMetrics.SKIPPED_TESTS, testsSkipped);
      context.saveMeasure(CoreMetrics.TEST_ERRORS, testsErrors);
      context.saveMeasure(CoreMetrics.TEST_FAILURES, testsFailures);
      context.saveMeasure(CoreMetrics.TEST_EXECUTION_TIME, testsTime);
    }
  }

  private void detailedMode(final SensorContext context, List<File> reports) throws XMLStreamException {
    for (File report : reports) {
      TestSuiteParser parserHandler = new TestSuiteParser();
      StaxParser parser = new StaxParser(parserHandler, false);
      parser.parse(report);

      LOG.info("Processing report '{}'", report);

      processReportDetailed(context, parserHandler.getParsedReports());
    }
  }

  private void processReportDetailed(SensorContext context, Collection<TestSuite> parsedReports) throws XMLStreamException {
    Collection<TestSuite> locatedResources = lookupResources(parsedReports);
    for (TestSuite fileReport : locatedResources) {
      InputFile inputFile = fileReport.getInputFile();

      LOG.debug("Saving test execution measures for '{}' under resource '{}'", fileReport.getKey(), inputFile.relativePath());

      context.saveMeasure(inputFile, CoreMetrics.SKIPPED_TESTS, (double) fileReport.getSkipped());
      context.saveMeasure(inputFile, CoreMetrics.TESTS, (double) fileReport.getTests() - fileReport.getSkipped());
      context.saveMeasure(inputFile, CoreMetrics.TEST_ERRORS, (double) fileReport.getErrors());
      context.saveMeasure(inputFile, CoreMetrics.TEST_FAILURES, (double) fileReport.getFailures());
      context.saveMeasure(inputFile, CoreMetrics.TEST_EXECUTION_TIME, (double) fileReport.getTime());

      double testsRun = (double)fileReport.getTests() - fileReport.getSkipped();
      if (testsRun > 0) {
        double passedTests = (double)fileReport.getTests() - fileReport.getErrors() - fileReport.getFailures() - fileReport.getSkipped();
        double successDensity = passedTests * PERCENT_BASE / testsRun;
        context.saveMeasure(inputFile, CoreMetrics.TEST_SUCCESS_DENSITY, ParsingUtils.scaleValue(successDensity));
      }
      context.saveMeasure(inputFile, new Measure(CoreMetrics.TEST_DATA, fileReport.getDetails()));
    }
  }

  private InputFile findResource(String fileKey) {
    return findResourceUsingNoseTestsStrategy(fileKey);
  }

  private InputFile findResourceUsingNoseTestsStrategy(String fileKey) {
    // a) check assuming the key doesnt contain the class name
    String candidateKey = StringUtils.replace(fileKey, ".", "/") + ".py";

    InputFile unitTestFile = getSonarTestFile(new File(candidateKey));

    if (unitTestFile == null) {
      // b) check assuming the key *does* contain the class name
      String candidateKey2 = StringUtils.replace(StringUtils.substringBeforeLast(fileKey, "."), ".", "/") + ".py";
      if ( !(candidateKey2.equals(candidateKey))) {
        unitTestFile = getSonarTestFile(new File(candidateKey2));
      }
    }

    return unitTestFile;
  }

  private Collection<TestSuite> lookupResources(Collection<TestSuite> testReports) {
    Map<String, TestSuite> locatedReports = new HashMap<>();

    for (TestSuite report : testReports) {
      String fileKey = report.getKey();

      LOG.debug("Trying to find a SonarQube resource for '{}' ...", fileKey);
      InputFile inputFile = findResource(fileKey);
      if (inputFile != null) {
        LOG.debug("... found! The resource is '{}'", inputFile);

        TestSuite summaryReport = locatedReports.get(inputFile.absolutePath());
        if (summaryReport != null) {
          summaryReport.addMeasures(report);
        } else {
          report.setInputFile(inputFile);
          locatedReports.put(inputFile.absolutePath(), report);
        }
      } else {
        LOG.debug("... cannot find the resource for '{}', drilling down to the details of this test wont be possible", fileKey);
      }
    }

    return locatedReports.values();
  }

  private InputFile getSonarTestFile(File file) {
    LOG.debug("Using the key '{}' to lookup the resource in SonarQube", file.getPath());
    return fileSystem.inputFile(fileSystem.predicates().is(file));
  }
}
