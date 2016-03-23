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
package org.sonar.plugins.python;

import com.google.common.collect.Lists;
import com.sonar.sslr.api.Grammar;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.issue.Issuable;
import org.sonar.api.issue.Issue;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.measures.PersistenceMode;
import org.sonar.api.measures.RangeDistributionBuilder;
import org.sonar.api.resources.Project;
import org.sonar.api.rule.RuleKey;
import org.sonar.python.PythonAstScanner;
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

import java.util.Collection;
import java.util.List;
import java.util.Locale;

public final class PythonSquidSensor implements Sensor {

  private static final Number[] FUNCTIONS_DISTRIB_BOTTOM_LIMITS = {1, 2, 4, 6, 8, 10, 12, 20, 30};
  private static final Number[] FILES_DISTRIB_BOTTOM_LIMITS = {0, 5, 10, 20, 30, 60, 90};

  private final Checks<SquidAstVisitor<Grammar>> checks;
  private final FileLinesContextFactory fileLinesContextFactory;

  private SensorContext context;
  private AstScanner<Grammar> scanner;
  private FileSystem fileSystem;
  private ResourcePerspectives resourcePerspectives;

  public PythonSquidSensor(FileLinesContextFactory fileLinesContextFactory, FileSystem fileSystem, ResourcePerspectives perspectives, CheckFactory checkFactory) {
    this.checks = checkFactory
        .<SquidAstVisitor<Grammar>>create(CheckList.REPOSITORY_KEY)
        .addAnnotatedChecks(CheckList.getChecks());
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.fileSystem = fileSystem;
    this.resourcePerspectives = perspectives;
  }

  @Override
  public boolean shouldExecuteOnProject(Project project) {
    FilePredicates p = fileSystem.predicates();
    return fileSystem.hasFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)));
  }

  @Override
  public void analyse(Project project, SensorContext context) {
    this.context = context;

    List<SquidAstVisitor<Grammar>> visitors = Lists.newArrayList(checks.all());
    visitors.add(new FileLinesVisitor(fileLinesContextFactory, fileSystem));
    this.scanner = PythonAstScanner.create(createConfiguration(), visitors.toArray(new SquidAstVisitor[visitors.size()]));
    FilePredicates p = fileSystem.predicates();
    scanner.scanFiles(Lists.newArrayList(fileSystem.files(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY)))));

    Collection<SourceCode> squidSourceFiles = scanner.getIndex().search(new QueryByType(SourceFile.class));
    save(squidSourceFiles);
  }

  private PythonConfiguration createConfiguration() {
    return new PythonConfiguration(fileSystem.encoding());
  }

  private void save(Collection<SourceCode> squidSourceFiles) {
    for (SourceCode squidSourceFile : squidSourceFiles) {
      SourceFile squidFile = (SourceFile) squidSourceFile;

      InputFile inputFile = fileSystem.inputFile(fileSystem.predicates().is(new java.io.File(squidFile.getKey())));

      saveFilesComplexityDistribution(inputFile, squidFile);
      saveFunctionsComplexityDistribution(inputFile, squidFile);
      saveMeasures(inputFile, squidFile);
      saveIssues(inputFile, squidFile);
    }
  }

  private void saveMeasures(InputFile sonarFile, SourceFile squidFile) {
    context.saveMeasure(sonarFile, CoreMetrics.FILES, squidFile.getDouble(PythonMetric.FILES));
    context.saveMeasure(sonarFile, CoreMetrics.LINES, squidFile.getDouble(PythonMetric.LINES));
    context.saveMeasure(sonarFile, CoreMetrics.NCLOC, squidFile.getDouble(PythonMetric.LINES_OF_CODE));
    context.saveMeasure(sonarFile, CoreMetrics.STATEMENTS, squidFile.getDouble(PythonMetric.STATEMENTS));
    context.saveMeasure(sonarFile, CoreMetrics.FUNCTIONS, squidFile.getDouble(PythonMetric.FUNCTIONS));
    context.saveMeasure(sonarFile, CoreMetrics.CLASSES, squidFile.getDouble(PythonMetric.CLASSES));
    context.saveMeasure(sonarFile, CoreMetrics.COMPLEXITY, squidFile.getDouble(PythonMetric.COMPLEXITY));
    context.saveMeasure(sonarFile, CoreMetrics.COMMENT_LINES, squidFile.getDouble(PythonMetric.COMMENT_LINES));
  }

  private void saveFunctionsComplexityDistribution(InputFile sonarFile, SourceFile squidFile) {
    Collection<SourceCode> squidFunctionsInFile = scanner.getIndex().search(new QueryByParent(squidFile), new QueryByType(SourceFunction.class));
    RangeDistributionBuilder complexityDistribution = new RangeDistributionBuilder(CoreMetrics.FUNCTION_COMPLEXITY_DISTRIBUTION, FUNCTIONS_DISTRIB_BOTTOM_LIMITS);
    for (SourceCode squidFunction : squidFunctionsInFile) {
      complexityDistribution.add(squidFunction.getDouble(PythonMetric.COMPLEXITY));
    }
    context.saveMeasure(sonarFile, complexityDistribution.build().setPersistenceMode(PersistenceMode.MEMORY));
  }

  private void saveFilesComplexityDistribution(InputFile sonarFile, SourceFile squidFile) {
    RangeDistributionBuilder complexityDistribution = new RangeDistributionBuilder(CoreMetrics.FILE_COMPLEXITY_DISTRIBUTION, FILES_DISTRIB_BOTTOM_LIMITS);
    complexityDistribution.add(squidFile.getDouble(PythonMetric.COMPLEXITY));
    context.saveMeasure(sonarFile, complexityDistribution.build().setPersistenceMode(PersistenceMode.MEMORY));
  }

  private void saveIssues(InputFile sonarFile, SourceFile squidFile) {
    Collection<CheckMessage> messages = squidFile.getCheckMessages();
    for (CheckMessage message : messages) {
      RuleKey ruleKey = checks.ruleKey((SquidAstVisitor<Grammar>) message.getCheck());
      Issuable issuable = resourcePerspectives.as(Issuable.class, sonarFile);

      if (issuable != null) {
        Issue issue = issuable.newIssueBuilder()
            .ruleKey(ruleKey)
            .line(message.getLine())
            .message(message.getText(Locale.ENGLISH))
            .build();
        issuable.addIssue(issue);
      }
    }
  }

  @Override
  public String toString() {
    return getClass().getSimpleName();
  }

}
