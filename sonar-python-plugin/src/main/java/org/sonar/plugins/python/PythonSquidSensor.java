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
package org.sonar.plugins.python;

import com.google.common.collect.Lists;
import com.sonar.sslr.api.Grammar;
import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.sonar.api.SonarProduct;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.ce.measure.RangeDistributionBuilder;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.measures.Metric;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.python.PythonAstScanner;
import org.sonar.python.PythonCheck;
import org.sonar.python.PythonCheck.IssueLocation;
import org.sonar.python.PythonCheck.PreciseIssue;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonMetric;
import org.sonar.python.checks.CheckList;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.squidbridge.AstScanner;
import org.sonar.squidbridge.SquidAstVisitor;
import org.sonar.squidbridge.api.CheckMessage;
import org.sonar.squidbridge.api.SourceCode;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.api.SourceFunction;
import org.sonar.squidbridge.indexer.QueryByParent;
import org.sonar.squidbridge.indexer.QueryByType;

public final class PythonSquidSensor implements Sensor {

  private static final Number[] FUNCTIONS_DISTRIB_BOTTOM_LIMITS = {1, 2, 4, 6, 8, 10, 12, 20, 30};
  private static final Number[] FILES_DISTRIB_BOTTOM_LIMITS = {0, 5, 10, 20, 30, 60, 90};

  private static final Version V6_0 = Version.create(6, 0);

  private final Checks<SquidAstVisitor<Grammar>> checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;

  private SensorContext context;
  private AstScanner<Grammar> scanner;

  public PythonSquidSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter) {
    this.checks = checkFactory
        .<SquidAstVisitor<Grammar>>create(CheckList.REPOSITORY_KEY)
        .addAnnotatedChecks(CheckList.getChecks());
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(Python.KEY)
      .name("Python Squid Sensor")
      .onlyOnFileType(Type.MAIN);
  }

  @Override
  public void execute(SensorContext context) {
    this.context = context;
    Map<InputFile, Set<Integer>> linesOfCode = new HashMap<>();

    PythonConfiguration conf = createConfiguration();

    List<SquidAstVisitor<Grammar>> visitors = Lists.newArrayList(checks.all());
    visitors.add(new FileLinesVisitor(fileLinesContextFactory, context.fileSystem(), linesOfCode, conf.getIgnoreHeaderComments()));
    visitors.add(new PythonHighlighter(context));
    scanner = PythonAstScanner.create(conf, visitors.toArray(new SquidAstVisitor[visitors.size()]));
    FilePredicates p = context.fileSystem().predicates();
    scanner.scanFiles(Lists.newArrayList(context.fileSystem().files(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)))));

    Collection<SourceCode> squidSourceFiles = scanner.getIndex().search(new QueryByType(SourceFile.class));
    save(squidSourceFiles);
    savePreciseIssues(
      visitors
        .stream()
        .filter(v -> v instanceof PythonCheck)
        .map(v -> (PythonCheck) v)
        .collect(Collectors.toList()));

    if (!isSonarLint(context)) {
      new PythonCoverageSensor().execute(context, linesOfCode);
    }
  }

  private void savePreciseIssues(List<PythonCheck> pythonChecks) {
    for (PythonCheck pythonCheck : pythonChecks) {
      RuleKey ruleKey = checks.ruleKey(pythonCheck);
      for (PreciseIssue preciseIssue : pythonCheck.getIssues()) {
        InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().is(preciseIssue.file()));

        NewIssue newIssue = context
          .newIssue()
          .forRule(ruleKey);

        Integer cost = preciseIssue.cost();
        if (cost != null) {
          newIssue.gap(cost.doubleValue());
        }

        newIssue.at(newLocation(inputFile, newIssue, preciseIssue.primaryLocation()));

        for (IssueLocation secondaryLocation : preciseIssue.secondaryLocations()) {
          newIssue.addLocation(newLocation(inputFile, newIssue, secondaryLocation));
        }

        newIssue.save();
      }
    }
  }

  private static NewIssueLocation newLocation(InputFile inputFile, NewIssue issue, IssueLocation location) {
    NewIssueLocation newLocation = issue.newLocation()
      .on(inputFile);
    if (location.startLine() != 0) {
      TextRange range = inputFile.newRange(
        location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset());
      newLocation.at(range);
    }

    if (location.message() != null) {
      newLocation.message(location.message());
    }
    return newLocation;
  }


  private PythonConfiguration createConfiguration() {
    return new PythonConfiguration(context.fileSystem().encoding());
  }

  private void save(Collection<SourceCode> squidSourceFiles) {
    for (SourceCode squidSourceFile : squidSourceFiles) {
      SourceFile squidFile = (SourceFile) squidSourceFile;

      InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().is(new java.io.File(squidFile.getKey())));

      noSonarFilter.noSonarInFile(inputFile, squidFile.getNoSonarTagLines());

      saveFilesComplexityDistribution(inputFile, squidFile);
      saveFunctionsComplexityDistribution(inputFile, squidFile);
      saveMeasures(inputFile, squidFile);
      saveIssues(inputFile, squidFile);
    }
  }

  private void saveMeasures(InputFile inputFile, SourceFile squidFile) {
    saveMetricOnFile(inputFile, CoreMetrics.NCLOC, squidFile.getInt(PythonMetric.LINES_OF_CODE));
    saveMetricOnFile(inputFile, CoreMetrics.STATEMENTS, squidFile.getInt(PythonMetric.STATEMENTS));
    saveMetricOnFile(inputFile, CoreMetrics.FUNCTIONS, squidFile.getInt(PythonMetric.FUNCTIONS));
    saveMetricOnFile(inputFile, CoreMetrics.CLASSES, squidFile.getInt(PythonMetric.CLASSES));
    saveMetricOnFile(inputFile, CoreMetrics.COMPLEXITY, squidFile.getInt(PythonMetric.COMPLEXITY));
    saveMetricOnFile(inputFile, CoreMetrics.COMMENT_LINES, squidFile.getInt(PythonMetric.COMMENT_LINES));
  }

  private <T extends Serializable> void saveMetricOnFile(InputFile inputFile, Metric metric, T value) {
    context.<T>newMeasure()
      .withValue(value)
      .forMetric(metric)
      .on(inputFile)
      .save();
  }

  private void saveFunctionsComplexityDistribution(InputFile inputFile, SourceFile squidFile) {
    Collection<SourceCode> squidFunctionsInFile = scanner.getIndex().search(new QueryByParent(squidFile), new QueryByType(SourceFunction.class));
    RangeDistributionBuilder complexityDistribution = new RangeDistributionBuilder(FUNCTIONS_DISTRIB_BOTTOM_LIMITS);
    for (SourceCode squidFunction : squidFunctionsInFile) {
      complexityDistribution.add(squidFunction.getDouble(PythonMetric.COMPLEXITY));
    }

    context.<String>newMeasure()
      .on(inputFile)
      .forMetric(CoreMetrics.FUNCTION_COMPLEXITY_DISTRIBUTION)
      .withValue(complexityDistribution.build())
      .save();
  }

  private void saveFilesComplexityDistribution(InputFile inputFile, SourceFile squidFile) {
    RangeDistributionBuilder complexityDistribution = new RangeDistributionBuilder(FILES_DISTRIB_BOTTOM_LIMITS);
    complexityDistribution.add(squidFile.getDouble(PythonMetric.COMPLEXITY));
    context.<String>newMeasure()
      .on(inputFile)
      .forMetric(CoreMetrics.FILE_COMPLEXITY_DISTRIBUTION)
      .withValue(complexityDistribution.build())
      .save();
  }

  private void saveIssues(InputFile inputFile, SourceFile squidFile) {
    Collection<CheckMessage> messages = squidFile.getCheckMessages();
    for (CheckMessage message : messages) {
      RuleKey ruleKey = checks.ruleKey((SquidAstVisitor<Grammar>) message.getCheck());
      NewIssue newIssue = context.newIssue();

      NewIssueLocation primaryLocation = newIssue.newLocation()
        .message(message.getText(Locale.ENGLISH))
        .on(inputFile);

      if (message.getLine() != null) {
        primaryLocation.at(inputFile.selectLine(message.getLine()));
      }

      newIssue.forRule(ruleKey).at(primaryLocation).save();
    }
  }

  private static boolean isSonarLint(SensorContext context) {
    return context.getSonarQubeVersion().isGreaterThanOrEqual(V6_0) && context.runtime().getProduct() == SonarProduct.SONARLINT;
  }

}
