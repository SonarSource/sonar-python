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
import org.sonar.api.batch.CoverageExtension;
import org.sonar.api.batch.DependsUpon;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.Qualifiers;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.api.utils.ParsingUtils;
import org.sonar.api.utils.StaxParser;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonReportSensor;

import java.io.File;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

@Properties({
  @Property(
    key = PythonXunitSensor.REPORT_PATH_KEY,
    defaultValue = PythonXunitSensor.DEFAULT_REPORT_PATH,
    name = "Path to xunit report(s)",
    description = "Path to the report of test execution, relative to project's root. Ant patterns are accepted. The reports have to conform to the junitreport XML format.",
    global = false,
    project = true)
})
public class PythonXunitSensor extends PythonReportSensor {

  public static final String REPORT_PATH_KEY = "sonar.python.xunit.reportPath";
  public static final String DEFAULT_REPORT_PATH = "xunit-reports/xunit-result-*.xml";
  private Python lang = null;

  public PythonXunitSensor(Settings conf, Python lang, ModuleFileSystem fileSystem) {
    super(conf, fileSystem);
    this.lang = lang;
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

  protected void processReport(final Project project, final SensorContext context, File report) throws javax.xml.stream.XMLStreamException {
    parseReport(project, context, report);
  }

  private void parseReport(Project project, SensorContext context, File report) throws javax.xml.stream.XMLStreamException {
    LOG.info("Parsing report '{}'", report);

    TestSuiteParser parserHandler = new TestSuiteParser();
    StaxParser parser = new StaxParser(parserHandler, false);
    parser.parse(report);

    Collection<TestSuite> locatedResources = lookupResources(project, context, parserHandler.getParsedReports());

    for (TestSuite fileReport : locatedResources) {
      org.sonar.api.resources.File unitTest = fileReport.getSonarResource();

      LOG.debug("Saving test execution measures for file '{}' under resource '{}'",
        fileReport.getKey(), unitTest);

      double testsCount = fileReport.getTests() - fileReport.getSkipped();
      context.saveMeasure(unitTest, CoreMetrics.SKIPPED_TESTS, (double) fileReport.getSkipped());
      context.saveMeasure(unitTest, CoreMetrics.TESTS, testsCount);
      context.saveMeasure(unitTest, CoreMetrics.TEST_ERRORS, (double) fileReport.getErrors());
      context.saveMeasure(unitTest, CoreMetrics.TEST_FAILURES, (double) fileReport.getFailures());
      context.saveMeasure(unitTest, CoreMetrics.TEST_EXECUTION_TIME, (double) fileReport.getTime());
      double passedTests = testsCount - fileReport.getErrors() - fileReport.getFailures();
      if (testsCount > 0) {
        double percentage = passedTests * 100d / testsCount;
        context.saveMeasure(unitTest, CoreMetrics.TEST_SUCCESS_DENSITY, ParsingUtils.scaleValue(percentage));
      }
      context.saveMeasure(unitTest, new Measure(CoreMetrics.TEST_DATA, fileReport.getDetails()));
    }
  }

  org.sonar.api.resources.File findResource(Project project, SensorContext context, String fileKey) {
    return findResourceUsingNosetestsStrategy(project, context, fileKey);
  }

  org.sonar.api.resources.File findResourceUsingNosetestsStrategy(Project project, SensorContext context, String fileKey) {
    // a) check assuming the key doesnt contain the class name
    String actualKey = StringUtils.replace(fileKey, ".", "/") + ".py";

    org.sonar.api.resources.File unitTestFile = getSonarTestFile(new File(actualKey), project);

    if (context.getResource(unitTestFile) == null) {
      // b) check assuming the key *does* contain the class name
      actualKey = StringUtils.replace(StringUtils.substringBeforeLast(fileKey, "."), ".", "/") + ".py";

      unitTestFile = getSonarTestFile(new File(actualKey), project);
      if (context.getResource(unitTestFile) == null) {
        unitTestFile = null;
      }
    }

    return unitTestFile;
  }

  private Collection<TestSuite> lookupResources(Project project, SensorContext context, Collection<TestSuite> testReports) {
    Map<String, TestSuite> locatedReports = new HashMap<String, TestSuite>();

    for (TestSuite report : testReports) {
      String fileKey = report.getKey();

      org.sonar.api.resources.File resource = findResource(project, context, fileKey);
      if (resource == null) {
        LOG.debug("Cannot find the resource for {}, creating a virtual one", fileKey);
        resource = createVirtualFile(context, fileKey);
      }

      TestSuite summaryReport = locatedReports.get(resource.getKey());
      if (summaryReport != null) {
        LOG.debug("Adding measures of {} to {}", summaryReport.getKey(), report.getKey());
        summaryReport.addMeasures(report);
      } else {
        report.setSonarResource(resource);
        locatedReports.put(resource.getKey(), report);
      }
    }

    return locatedReports.values();
  }

  private org.sonar.api.resources.File createVirtualFile(SensorContext context,
                                                         String filename) {
    org.sonar.api.resources.File virtualFile = new org.sonar.api.resources.File(this.lang, filename);
    virtualFile.setQualifier(Qualifiers.UNIT_TEST_FILE);
    context.saveSource(virtualFile, "<source code could not be found>");
    return virtualFile;
  }

  private org.sonar.api.resources.File getSonarTestFile(File file, Project project) {
    org.sonar.api.resources.File unitTestFile = org.sonar.api.resources.File.fromIOFile(file, project);

    if (unitTestFile == null) {
      unitTestFile = org.sonar.api.resources.File.fromIOFile(file, fileSystem.testDirs());
    }
    return unitTestFile;
  }
}
