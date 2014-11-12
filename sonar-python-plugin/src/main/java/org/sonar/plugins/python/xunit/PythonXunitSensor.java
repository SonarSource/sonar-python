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
package org.sonar.plugins.python.xunit;

import org.apache.commons.lang.StringUtils;
import org.sonar.api.Properties;
import org.sonar.api.Property;
import org.sonar.api.PropertyType;
import org.sonar.api.batch.CoverageExtension;
import org.sonar.api.batch.DependsUpon;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.api.utils.ParsingUtils;
import org.sonar.api.utils.StaxParser;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonReportSensor;

import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.List;

@Properties({
  @Property(
    key = PythonXunitSensor.REPORT_PATH_KEY,
    defaultValue = PythonXunitSensor.DEFAULT_REPORT_PATH,
    name = "Path to xunit report(s)",
    description = "Path to the report of test execution, relative to project's root. Ant patterns are accepted. The reports have to conform to the junitreport XML format.",
    global = false,
    project = true),

  @Property(
    key = PythonXunitSensor.SKIP_DETAILS,
    type = PropertyType.BOOLEAN,
    defaultValue = "true",
    name = "Skip the details when importing the Xunit reports",
    description = "If 'true', provides the test execution statistics only on project level, but makes the import procedure more mature",
    global = false,
    project = true)
})
public class PythonXunitSensor extends PythonReportSensor {

  public static final String REPORT_PATH_KEY = "sonar.python.xunit.reportPath";
  public static final String DEFAULT_REPORT_PATH = "xunit-reports/xunit-result-*.xml";
  public static final String SKIP_DETAILS = "sonar.python.xunit.skipDetails";

  private ResourceFinder resourceFinder = null;

  public PythonXunitSensor(Settings conf, ModuleFileSystem fileSystem) {
    super(conf, fileSystem);
    this.resourceFinder = new DefaultResourceFinder();
  }

  void injectResourceFinder(ResourceFinder finder){
    this.resourceFinder = finder;
  }

  @DependsUpon
  public Class dependsUponCoverageSensors() {
    return CoverageExtension.class;
  }

  protected String reportPathKey() {
    return REPORT_PATH_KEY;
  }

  protected String defaultReportPath() {
    return DEFAULT_REPORT_PATH;
  }

  protected void processReports(final Project project, final SensorContext context, List<File> reports)
    throws javax.xml.stream.XMLStreamException {

    if(conf.getBoolean(SKIP_DETAILS)){
      simpleMode(project, context, reports);
    }
    else{
      detailledMode(project, context, reports);
    }
  }

  private void simpleMode(final Project project, final SensorContext context, List<File> reports)
    throws javax.xml.stream.XMLStreamException
  {
    TestSuiteParser parserHandler = new TestSuiteParser();
    StaxParser parser = new StaxParser(parserHandler, false);
    for (File report: reports){
      parser.parse(report);
    }

    double testsCount = 0.0;
    double testsSkipped = 0.0;
    double testsErrors = 0.0;
    double testsFailures = 0.0;
    double testsTime = 0.0;
    for(TestSuite report: parserHandler.getParsedReports()){
      testsCount += report.getTests() - report.getSkipped();
      testsSkipped += report.getSkipped();
      testsErrors += report.getErrors();
      testsFailures += report.getFailures();
      testsTime += report.getTime();
    }

    if (testsCount > 0) {
      double testsPassed = testsCount - testsErrors - testsFailures;
      double successDensity = testsPassed * 100d / testsCount;
      context.saveMeasure(project, CoreMetrics.TEST_SUCCESS_DENSITY,
                          ParsingUtils.scaleValue(successDensity));

      context.saveMeasure(project, CoreMetrics.TESTS, testsCount);
      context.saveMeasure(project, CoreMetrics.SKIPPED_TESTS, testsSkipped);
      context.saveMeasure(project, CoreMetrics.TEST_ERRORS, testsErrors);
      context.saveMeasure(project, CoreMetrics.TEST_FAILURES, testsFailures);
      context.saveMeasure(project, CoreMetrics.TEST_EXECUTION_TIME, testsTime);
    }
  }

  private void detailledMode(final Project project, final SensorContext context,
                             List<File> reports)
    throws javax.xml.stream.XMLStreamException {
    for(File report: reports){
      TestSuiteParser parserHandler = new TestSuiteParser();
      StaxParser parser = new StaxParser(parserHandler, false);
      parser.parse(report);

      LOG.info("Processing report '{}'", report);

      processReportDetailled(project, context, parserHandler.getParsedReports());
    }
  }

  private void processReportDetailled(Project project, SensorContext context, Collection<TestSuite> parsedReports)
    throws javax.xml.stream.XMLStreamException
  {
    Collection<TestSuite> locatedResources = lookupResources(project, context, parsedReports);
    for (TestSuite fileReport : locatedResources) {
      org.sonar.api.resources.File unitTest = fileReport.getSonarResource();

      LOG.debug("Saving test execution measures for '{}' under resource '{}'",
                fileReport.getKey(), unitTest.getKey());

      context.saveMeasure(unitTest, CoreMetrics.SKIPPED_TESTS, (double) fileReport.getSkipped());
      context.saveMeasure(unitTest, CoreMetrics.TESTS, (double) fileReport.getTests() - fileReport.getSkipped());
      context.saveMeasure(unitTest, CoreMetrics.TEST_ERRORS, (double) fileReport.getErrors());
      context.saveMeasure(unitTest, CoreMetrics.TEST_FAILURES, (double) fileReport.getFailures());
      context.saveMeasure(unitTest, CoreMetrics.TEST_EXECUTION_TIME, (double) fileReport.getTime());

      double testsRun = fileReport.getTests() - fileReport.getSkipped();
      if (testsRun > 0) {
        double passedTests = fileReport.getTests() - fileReport.getErrors() - fileReport.getFailures() - fileReport.getSkipped();
        double successDensity = passedTests * 100d / testsRun;
        context.saveMeasure(unitTest, CoreMetrics.TEST_SUCCESS_DENSITY, ParsingUtils.scaleValue(successDensity));
      }
      context.saveMeasure(unitTest, new Measure(CoreMetrics.TEST_DATA, fileReport.getDetails()));
    }
  }

  org.sonar.api.resources.File findResource(Project project, SensorContext context, String fileKey) {
    return findResourceUsingNosetestsStrategy(project, context, fileKey);
  }

  org.sonar.api.resources.File findResourceUsingNosetestsStrategy(Project project, SensorContext context, String fileKey) {
    // a) check assuming the key doesnt contain the class name
    String candidateKey = StringUtils.replace(fileKey, ".", "/") + ".py";

    org.sonar.api.resources.File unitTestFile = getSonarTestFile(new File(candidateKey), context, project);

    if (unitTestFile == null) {
      // b) check assuming the key *does* contain the class name
      String candidateKey2 = StringUtils.replace(StringUtils.substringBeforeLast(fileKey, "."), ".", "/") + ".py";
      if(!(candidateKey2.equals(candidateKey))){
        unitTestFile = getSonarTestFile(new File(candidateKey2), context, project);
      }
    }

    return unitTestFile;
  }

  private Collection<TestSuite> lookupResources(Project project, SensorContext context, Collection<TestSuite> testReports) {
    Map<String, TestSuite> locatedReports = new HashMap<String, TestSuite>();

    for (TestSuite report : testReports) {
      String fileKey = report.getKey();

      LOG.debug("Trying to find a SonarQube resource for '{}' ...", fileKey);
      org.sonar.api.resources.File resource = findResource(project, context, fileKey);
      if (resource != null) {
        LOG.debug("... found! The resource is '{}'", resource);

        TestSuite summaryReport = locatedReports.get(resource.getKey());
        if (summaryReport != null) {
          summaryReport.addMeasures(report);
        } else {
          report.setSonarResource(resource);
          locatedReports.put(resource.getKey(), report);
        }
      } else {
        LOG.debug("... cannot find the resource for '{}', drilling down to the details of this test wont be possible", fileKey);
      }
    }

    return locatedReports.values();
  }

  private org.sonar.api.resources.File getSonarTestFile(File file, SensorContext context, Project project) {
    LOG.debug("Using the key '{}' to lookup the resource in SonarQube", file.getPath());
    return resourceFinder.findInSonar(file, context, fileSystem, project);
  }
}
