/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.api.SonarProduct;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.issue.NewIssue;
import org.sonar.api.batch.sensor.issue.NewIssueLocation;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.measures.Metric;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.Version;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.cpd.PythonCpdAnalyzer;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.python.metrics.FileMetrics;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonarsource.sonarlint.plugin.api.SonarLintRuntime;
import org.sonarsource.sonarlint.plugin.api.issue.NewInputFileEdit;
import org.sonarsource.sonarlint.plugin.api.issue.NewQuickFix;
import org.sonarsource.sonarlint.plugin.api.issue.NewSonarLintIssue;

public class PythonScanner extends Scanner {

  private static final Logger LOG = Loggers.get(PythonScanner.class);

  private final PythonParser parser;
  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;
  private final PythonCpdAnalyzer cpdAnalyzer;
  private final PythonIndexer indexer;

  public PythonScanner(
    SensorContext context, PythonChecks checks,
    FileLinesContextFactory fileLinesContextFactory, NoSonarFilter noSonarFilter, PythonIndexer indexer) {
    super(context);
    this.checks = checks;
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.cpdAnalyzer = new PythonCpdAnalyzer(context);
    this.parser = PythonParser.create();
    this.indexer = indexer;
    this.indexer.buildOnce(context);
  }

  @Override
  protected String name() {
    return "rules execution";
  }

  @Override
  protected void scanFile(InputFile inputFile) {
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile);
    PythonVisitorContext visitorContext;
    try {
      AstNode astNode = parser.parse(pythonFile.content());
      FileInput parse = new PythonTreeMaker().fileInput(astNode);
      visitorContext = new PythonVisitorContext(parse, pythonFile, getWorkingDirectory(context), indexer.packageName(inputFile), indexer.projectLevelSymbolTable());
      if (!isInSonarLint(context)) {
        saveMeasures(inputFile, visitorContext);
      }
    } catch (RecognitionException e) {
      visitorContext = new PythonVisitorContext(pythonFile, e);
      LOG.error("Unable to parse file: " + inputFile);
      LOG.error(e.getMessage());
      context.newAnalysisError()
        .onFile(inputFile)
        .at(inputFile.newPointer(e.getLine(), 0))
        .message(e.getMessage())
        .save();
    }
    List<PythonSubscriptionCheck> checksBasedOnTree = new ArrayList<>();
    for (PythonCheck check : checks.all()) {
      if (check instanceof PythonSubscriptionCheck) {
        checksBasedOnTree.add((PythonSubscriptionCheck) check);
      } else {
        check.scanFile(visitorContext);
      }
    }
    SubscriptionVisitor.analyze(checksBasedOnTree, visitorContext);
    saveIssues(inputFile, visitorContext.getIssues());

    if (visitorContext.rootTree() != null && !isInSonarLint(context)) {
      new SymbolVisitor(context.newSymbolTable().onFile(inputFile)).visitFileInput(visitorContext.rootTree());
      new PythonHighlighter(context, inputFile).scanFile(visitorContext);
    }
  }

  // visible for testing
  static File getWorkingDirectory(SensorContext context) {
    return isInSonarLint(context) ? null : context.fileSystem().workDir();
  }

  private static boolean isInSonarLint(SensorContext context) {
    return context.runtime().getProduct().equals(SonarProduct.SONARLINT);
  }

  public boolean isQuickFixCompatible(SensorContext context) {
    return isInSonarLint(context) && ((SonarLintRuntime) context.runtime()).getSonarLintPluginApiVersion().isGreaterThanOrEqual(Version.parse("6.3"));
  }

  @Override
  protected void processException(Exception e, InputFile file) {
    LOG.warn("Unable to analyze file: " + file, e);
  }

  private void saveIssues(InputFile inputFile, List<PreciseIssue> issues) {
    for (PreciseIssue preciseIssue : issues) {
      RuleKey ruleKey = checks.ruleKey(preciseIssue.check());
      NewIssue newIssue = context
        .newIssue()
        .forRule(ruleKey);

      Integer cost = preciseIssue.cost();
      if (cost != null) {
        newIssue.gap(cost.doubleValue());
      }

      NewIssueLocation primaryLocation = newLocation(inputFile, newIssue, preciseIssue.primaryLocation());
      newIssue.at(primaryLocation);

      Deque<NewIssueLocation> secondaryLocationsFlow = new ArrayDeque<>();

      for (IssueLocation secondaryLocation : preciseIssue.secondaryLocations()) {
        String fileId = secondaryLocation.fileId();
        if (fileId != null) {
          InputFile issueLocationFile = component(fileId, context);
          if (issueLocationFile != null) {
            secondaryLocationsFlow.addFirst(newLocation(issueLocationFile, newIssue, secondaryLocation));
          }
        } else {
          newIssue.addLocation(newLocation(inputFile, newIssue, secondaryLocation));
        }
      }

      // secondary locations on multiple files are only supported using flows
      if (!secondaryLocationsFlow.isEmpty()) {
        secondaryLocationsFlow.addFirst(primaryLocation);
        newIssue.addFlow(secondaryLocationsFlow);
      }

      handleQuickFixes(inputFile, ruleKey, newIssue, preciseIssue);

      newIssue.save();
    }
  }

  @CheckForNull
  private InputFile component(String fileId, SensorContext sensorContext) {
    InputFile inputFile = Optional.ofNullable(sensorContext.fileSystem().inputFile(sensorContext.fileSystem().predicates().is(new File(fileId))))
      .orElseGet(() -> indexer.getFileWithId(fileId));
    if (inputFile == null) {
      LOG.debug("Failed to find InputFile for {}", fileId);
    }
    return inputFile;
  }

  private static NewIssueLocation newLocation(InputFile inputFile, NewIssue issue, IssueLocation location) {
    NewIssueLocation newLocation = issue.newLocation()
      .on(inputFile);
    if (location.startLine() != IssueLocation.UNDEFINED_LINE) {
      TextRange range;
      if (location.startLineOffset() == IssueLocation.UNDEFINED_OFFSET) {
        range = inputFile.selectLine(location.startLine());
      } else {
        range = inputFile.newRange(location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset());
      }
      newLocation.at(range);
    }

    String message = location.message();
    if (message != null) {
      newLocation.message(message);
    }
    return newLocation;
  }

  private void saveMeasures(InputFile inputFile, PythonVisitorContext visitorContext) {
    FileMetrics fileMetrics = new FileMetrics(visitorContext);
    FileLinesVisitor fileLinesVisitor = fileMetrics.fileLinesVisitor();

    cpdAnalyzer.pushCpdTokens(inputFile, visitorContext);
    noSonarFilter.noSonarInFile(inputFile, fileLinesVisitor.getLinesWithNoSonar());

    Set<Integer> linesOfCode = fileLinesVisitor.getLinesOfCode();
    saveMetricOnFile(inputFile, CoreMetrics.NCLOC, linesOfCode.size());
    saveMetricOnFile(inputFile, CoreMetrics.STATEMENTS, fileMetrics.numberOfStatements());
    saveMetricOnFile(inputFile, CoreMetrics.FUNCTIONS, fileMetrics.numberOfFunctions());
    saveMetricOnFile(inputFile, CoreMetrics.CLASSES, fileMetrics.numberOfClasses());
    saveMetricOnFile(inputFile, CoreMetrics.COMPLEXITY, fileMetrics.complexity());
    saveMetricOnFile(inputFile, CoreMetrics.COGNITIVE_COMPLEXITY, fileMetrics.cognitiveComplexity());
    saveMetricOnFile(inputFile, CoreMetrics.COMMENT_LINES, fileLinesVisitor.getCommentLineCount());

    FileLinesContext fileLinesContext = fileLinesContextFactory.createFor(inputFile);
    for (int line : linesOfCode) {
      fileLinesContext.setIntValue(CoreMetrics.NCLOC_DATA_KEY, line, 1);
    }
    for (int line : fileLinesVisitor.getExecutableLines()) {
      fileLinesContext.setIntValue(CoreMetrics.EXECUTABLE_LINES_DATA_KEY, line, 1);
    }
    fileLinesContext.save();
  }

  private void saveMetricOnFile(InputFile inputFile, Metric<Integer> metric, Integer value) {
    context.<Integer>newMeasure()
      .withValue(value)
      .forMetric(metric)
      .on(inputFile)
      .save();
  }

  private void handleQuickFixes(InputFile inputFile, RuleKey ruleKey, NewIssue newIssue, PreciseIssue preciseIssue) {
    if (newIssue instanceof NewSonarLintIssue && preciseIssue instanceof IssueWithQuickFix) {
      List<PythonQuickFix> quickFixes = ((IssueWithQuickFix) preciseIssue).getQuickFixes();
        addQuickFixes(inputFile, ruleKey, quickFixes, (NewSonarLintIssue) newIssue);
    }
  }

  private static void addQuickFixes(InputFile inputFile, RuleKey ruleKey, Iterable<PythonQuickFix> quickFixes, NewSonarLintIssue sonarLintIssue) {
    try {
      for (PythonQuickFix quickFix : quickFixes) {
        NewQuickFix newQuickFix = sonarLintIssue.newQuickFix()
          .message(quickFix.getDescription());

        NewInputFileEdit edit = newQuickFix.newInputFileEdit().on(inputFile);

        quickFix.getTextEdits().stream()
          .map(pythonTextEdit -> edit.newTextEdit().at(rangeFromTextSpan(inputFile, pythonTextEdit))
            .withNewText(pythonTextEdit.replacementText()))
          .forEach(edit::addTextEdit);
        newQuickFix.addInputFileEdit(edit);
        sonarLintIssue.addQuickFix(newQuickFix);
      }
      // TODO : is this try/catch still necessary ?
    } catch (RuntimeException e) {
      // We still want to report the issue if we did not manage to create a quick fix.
      LOG.warn(String.format("Could not report quick fixes for rule: %s. %s: %s", ruleKey, e.getClass().getName(), e.getMessage()));
    }
  }

  private static TextRange rangeFromTextSpan(InputFile file, PythonTextEdit pythonTextEdit) {
    IssueLocation issue = pythonTextEdit.issueLocation;
    return file.newRange(issue.startLine(), issue.startLineOffset(), issue.endLine(), issue.endLineOffset());
  }
}
