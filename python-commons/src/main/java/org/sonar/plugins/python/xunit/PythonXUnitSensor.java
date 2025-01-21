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
package org.sonar.plugins.python.xunit;

import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.CheckForNull;
import javax.xml.stream.XMLStreamException;
import org.apache.commons.lang.StringUtils;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputComponent;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.measure.Metric;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;
import org.sonar.api.measures.CoreMetrics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.PythonReportSensor;
import org.sonar.plugins.python.parser.StaxParser;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;

public class PythonXUnitSensor extends PythonReportSensor {
  private static final Logger LOG = LoggerFactory.getLogger(PythonXUnitSensor.class);

  public static final String REPORT_PATH_KEY = "sonar.python.xunit.reportPath";
  public static final String DEFAULT_REPORT_PATH = "xunit-reports/xunit-result-*.xml";
  public static final String SKIP_DETAILS = "sonar.python.xunit.skipDetails";

  private final FileSystem fileSystem;

  public PythonXUnitSensor(Configuration conf, FileSystem fileSystem, AnalysisWarningsWrapper analysisWarnings) {
    super(conf, analysisWarnings, "XUnit");
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

    TestResult total = new TestResult();
    parserHandler.getParsedReports().forEach(testSuite -> testSuite.getTestCases().forEach(total::addTestCase));

    if (total.getTests() > 0) {
      InputComponent module = context.module();
      saveMeasure(context, module, CoreMetrics.TESTS, total.getExecutedTests());
      saveMeasure(context, module, CoreMetrics.SKIPPED_TESTS, total.getSkipped());
      saveMeasure(context, module, CoreMetrics.TEST_ERRORS, total.getErrors());
      saveMeasure(context, module, CoreMetrics.TEST_FAILURES, total.getFailures());
      saveMeasure(context, module, CoreMetrics.TEST_EXECUTION_TIME, total.getTime());
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
    Map<InputFile, TestResult> locatedResources = lookupResources(parsedReports);
    for (Map.Entry<InputFile, TestResult> entry : locatedResources.entrySet()) {
      InputFile inputFile = entry.getKey();
      TestResult fileTestResult = entry.getValue();
      LOG.debug("Saving test execution measures for '{}'", inputFile);

      saveMeasure(context, inputFile, CoreMetrics.SKIPPED_TESTS, fileTestResult.getSkipped());
      saveMeasure(context, inputFile, CoreMetrics.TESTS, fileTestResult.getExecutedTests());
      saveMeasure(context, inputFile, CoreMetrics.TEST_ERRORS, fileTestResult.getErrors());
      saveMeasure(context, inputFile, CoreMetrics.TEST_FAILURES, fileTestResult.getFailures());
      saveMeasure(context, inputFile, CoreMetrics.TEST_EXECUTION_TIME, fileTestResult.getTime());
    }
  }

  @CheckForNull
  private InputFile findResource(TestCase testCase, String fileKey) {
    InputFile unitTestFile = null;

    if (testCase.getFile() != null) {
      unitTestFile = getSonarTestFile(new File(testCase.getFile()));
    }

    if (unitTestFile == null) {
      String testClassname = testCase.getTestClassname();
      String key = testClassname != null ? testClassname : fileKey;
      return findResourceUsingNoseTestsStrategy(key);
    }

    return unitTestFile;
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

  private Map<InputFile, TestResult> lookupResources(Collection<TestSuite> testReports) {
    Map<InputFile, TestResult> testResultsByFile = new HashMap<>();

    for (TestSuite testSuite : testReports) {
      testSuite.getTestCases().forEach(testCase -> {
        String testClassname = testCase.getTestClassname();
        LOG.debug("Trying to find a SonarQube resource for test case '{}'", testClassname);
        InputFile inputFile = findResource(testCase, testSuite.getKey());
        if (inputFile != null) {
          LOG.debug("The resource was found '{}'", inputFile);
          testResultsByFile.computeIfAbsent(inputFile, k -> new TestResult()).addTestCase(testCase);
        } else {
          LOG.warn("The resource for '{}' is not found, drilling down to the details of this test won't be possible", testClassname);
        }
      });
    }

    return testResultsByFile;
  }

  @CheckForNull
  private InputFile getSonarTestFile(File file) {
    LOG.debug("Using the key '{}' to lookup the resource in SonarQube", file.getPath());
    var predicate = file.isAbsolute() ? fileSystem.predicates().hasAbsolutePath(file.getAbsolutePath())
      : fileSystem.predicates().hasRelativePath(file.getPath());

    return fileSystem.inputFile(predicate);
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
