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
package org.sonar.plugins.python.xunit;

import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.xml.stream.XMLStreamException;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputComponent;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.measure.Metric;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.plugins.python.PythonReportSensor;
import org.sonar.plugins.python.parser.StaxParser;

public class PythonXUnitSensor extends PythonReportSensor {
  private static final Logger LOG = LoggerFactory.getLogger(PythonXUnitSensor.class);

  public static final String REPORT_PATH_KEY = "sonar.python.xunit.reportPath";
  public static final String DEFAULT_REPORT_PATH = "xunit-reports/xunit-result-*.xml";
  public static final String SKIP_DETAILS = "sonar.python.xunit.skipDetails";

  private final FileSystem fileSystem;

  public PythonXUnitSensor(Configuration conf, FileSystem fileSystem) {
    super(conf);
    this.fileSystem = fileSystem;
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
    if (conf.getBoolean(SKIP_DETAILS).orElse(Boolean.FALSE)) {
      simpleMode(context, reports);
    } else {
      detailedMode(context, reports);
    }
  }

  private static void simpleMode(final SensorContext context, List<File> reports) throws XMLStreamException {
    TestSuiteParser parserHandler = new TestSuiteParser();
    StaxParser parser = new StaxParser(parserHandler);
    for (File report : reports) {
      parser.parse(report);
    }

    int testsCount = 0;
    int testsSkipped = 0;
    int testsErrors = 0;
    int testsFailures = 0;
    long testsTime = 0;
    for (TestSuite report : parserHandler.getParsedReports()) {
      testsCount += report.getTests() - report.getSkipped();
      testsSkipped += report.getSkipped();
      testsErrors += report.getErrors();
      testsFailures += report.getFailures();
      testsTime += report.getTime();
    }

    if (testsCount > 0) {
      InputComponent module = context.module();
      saveMeasure(context, module, CoreMetrics.TESTS, testsCount);
      saveMeasure(context, module, CoreMetrics.SKIPPED_TESTS, testsSkipped);
      saveMeasure(context, module, CoreMetrics.TEST_ERRORS, testsErrors);
      saveMeasure(context, module, CoreMetrics.TEST_FAILURES, testsFailures);
      saveMeasure(context, module, CoreMetrics.TEST_EXECUTION_TIME, testsTime);
    }
  }

  private void detailedMode(final SensorContext context, List<File> reports) throws XMLStreamException {
    for (File report : reports) {
      TestSuiteParser parserHandler = new TestSuiteParser();
      StaxParser parser = new StaxParser(parserHandler);
      parser.parse(report);

      LOG.info("Processing report '{}'", report);

      processReportDetailed(context, parserHandler.getParsedReports());
    }
  }

  private void processReportDetailed(SensorContext context, Collection<TestSuite> parsedReports) {
    Collection<TestSuite> locatedResources = lookupResources(parsedReports);
    for (TestSuite fileReport : locatedResources) {
      InputFile inputFile = fileReport.getInputFile();

      if (LOG.isDebugEnabled()) {
        LOG.debug("Saving test execution measures for '{}' under resource '{}'", fileReport.getKey(), inputFile.relativePath());
      }

      saveMeasure(context, inputFile, CoreMetrics.SKIPPED_TESTS, fileReport.getSkipped());
      saveMeasure(context, inputFile, CoreMetrics.TESTS, fileReport.getTests() - fileReport.getSkipped());
      saveMeasure(context, inputFile, CoreMetrics.TEST_ERRORS, fileReport.getErrors());
      saveMeasure(context, inputFile, CoreMetrics.TEST_FAILURES, fileReport.getFailures());
      saveMeasure(context, inputFile, CoreMetrics.TEST_EXECUTION_TIME, fileReport.getTime());
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

      LOG.debug("Trying to find a SonarQube resource for '{}'", fileKey);
      InputFile inputFile = findResource(fileKey);
      if (inputFile != null) {
        LOG.debug("The resource was found '{}'", inputFile);

        TestSuite summaryReport = locatedReports.get(inputFile.absolutePath());
        if (summaryReport != null) {
          summaryReport.addMeasures(report);
        } else {
          report.setInputFile(inputFile);
          locatedReports.put(inputFile.absolutePath(), report);
        }
      } else {
        LOG.warn("The resource for '{}' is not found, drilling down to the details of this test won't be possible", fileKey);
      }
    }

    return locatedReports.values();
  }

  private InputFile getSonarTestFile(File file) {
    LOG.debug("Using the key '{}' to lookup the resource in SonarQube", file.getPath());
    return fileSystem.inputFile(fileSystem.predicates().is(file));
  }

  private static void saveMeasure(SensorContext context, InputComponent component, Metric<Integer> metric, int value) {
    context.<Integer>newMeasure()
      .on(component)
      .forMetric(metric)
      .withValue(value)
      .save();
  }

  private static void saveMeasure(SensorContext context, InputComponent component, Metric<Long> metric, long value) {
    context.<Long>newMeasure()
      .on(component)
      .forMetric(metric)
      .withValue(value)
      .save();
  }

}
