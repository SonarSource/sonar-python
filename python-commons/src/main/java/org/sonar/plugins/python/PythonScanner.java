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
package org.sonar.plugins.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonFileConsumer;
import org.sonar.plugins.python.api.PythonInputFileContext;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.cpd.PythonCpdAnalyzer;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.python.IPythonLocation;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.python.metrics.FileMetrics;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.IPythonTreeMaker;
import org.sonar.python.tree.PythonTreeMaker;

public class PythonScanner extends Scanner {

  private static final Logger LOG = LoggerFactory.getLogger(PythonScanner.class);
  private static final Pattern DATABRICKS_MAGIC_COMMAND_PATTERN = Pattern.compile("^\\h*#\\h*(MAGIC|COMMAND).*");
  public static final String THREADS_PROPERTY_NAME = "sonar.python.analysis.threads";
  private static final String ARCHITECTURE_CALLBACK_LOCK_KEY = "architectureCallbackLock";

  private final Supplier<PythonParser> parserSupplier;
  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;
  private final PythonCpdAnalyzer cpdAnalyzer;
  private final PythonIndexer indexer;
  private final Map<PythonInputFile, Set<Class<? extends PythonCheck>>> checksExecutedWithoutParsingByFiles;
  private final AtomicInteger recognitionErrorCount;
  private final AtomicBoolean foundDatabricks;
  private final PythonFileConsumer architectureCallback;
  private final Map<String, Object> repositoryLocks;

  public PythonScanner(
    SensorContext context, PythonChecks checks, FileLinesContextFactory fileLinesContextFactory, NoSonarFilter noSonarFilter,
    Supplier<PythonParser> parserSupplier, PythonIndexer indexer, PythonFileConsumer architectureCallback) {
    super(context);
    this.checks = checks;
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.cpdAnalyzer = new PythonCpdAnalyzer(context);
    this.parserSupplier = parserSupplier;
    this.indexer = indexer;
    this.indexer.buildOnce(context);
    this.architectureCallback = architectureCallback;
    this.checksExecutedWithoutParsingByFiles = new ConcurrentHashMap<>();
    this.recognitionErrorCount = new AtomicInteger(0);
    this.foundDatabricks = new AtomicBoolean(false);
    this.repositoryLocks = new ConcurrentHashMap<>();
  }

  @Override
  protected String name() {
    return "rules execution";
  }

  @Override
  protected void processFiles(List<PythonInputFile> files, SensorContext context, MultiFileProgressReport progressReport,
    AtomicInteger numScannedWithoutParsing) {
    var numberOfThreads = getNumberOfThreads(context);
    if (numberOfThreads == 1) {
      super.processFiles(files, context, progressReport, numScannedWithoutParsing);
      return;
    }
    var pool = new ForkJoinPool(numberOfThreads);
    try {
      LOG.debug("Scanning files in {} threads", numberOfThreads);
      pool.submit(() -> super.processFiles(files, context, progressReport, numScannedWithoutParsing))
        .join();
    } finally {
      pool.shutdown();
    }
  }

  @Override
  protected Stream<PythonInputFile> getFilesStream(List<PythonInputFile> files) {
    return getNumberOfThreads(context) == 1 ? files.stream() : files.stream().parallel();
  }

  @Override
  protected void scanFile(PythonInputFile inputFile) throws IOException {
    var pythonFile = SonarQubePythonFile.create(inputFile);
    InputFile.Type fileType = inputFile.wrappedFile().type();
    PythonVisitorContext visitorContext = createVisitorContext(inputFile, pythonFile, fileType);

    executeChecks(visitorContext, checks.sonarPythonChecks(), fileType, inputFile);
    executeOtherChecks(inputFile, visitorContext, fileType);

    synchronized (repositoryLocks.computeIfAbsent(ARCHITECTURE_CALLBACK_LOCK_KEY, k -> new Object())) {
      architectureCallback.scanFile(visitorContext);
    }

    saveIssues(inputFile, visitorContext.getIssues());

    if (visitorContext.rootTree() != null && !isInSonarLint(context)) {
      new SymbolVisitor(context.newSymbolTable().onFile(inputFile.wrappedFile())).visitFileInput(visitorContext.rootTree());
      new PythonHighlighter(context, inputFile).scanFile(visitorContext);
    }

    searchForDataBricks(visitorContext);
  }


  private PythonVisitorContext createVisitorContext(PythonInputFile inputFile, PythonFile pythonFile, InputFile.Type fileType) throws IOException {
    PythonVisitorContext visitorContext;
    try {
      AstNode astNode = parserSupplier.get().parse(inputFile.contents());
      PythonTreeMaker treeMaker = getTreeMaker(inputFile);
      FileInput parse = treeMaker.fileInput(astNode);
      visitorContext = new PythonVisitorContext(parse,
        pythonFile,
        getWorkingDirectory(context),
        indexer.packageName(inputFile),
        indexer.projectLevelSymbolTable(),
        indexer.cacheContext(),
        context.runtime().getProduct());
      if (fileType == InputFile.Type.MAIN) {
        saveMeasures(inputFile, visitorContext);
      }
    } catch (RecognitionException e) {
      visitorContext = new PythonVisitorContext(pythonFile, e, context.runtime().getProduct());

      var line = (inputFile.kind() == PythonInputFile.Kind.IPYTHON) ?
        ((GeneratedIPythonFile) inputFile).locationMap().get(e.getLine()).line() : e.getLine();
      var newMessage = e.getMessage().replace("line " + e.getLine(), "line " + line);

      LOG.error("Unable to parse file: {}", inputFile);
      LOG.error(newMessage);
      recognitionErrorCount.incrementAndGet();
      context.newAnalysisError()
        .onFile(inputFile.wrappedFile())
        .at(inputFile.wrappedFile().newPointer(line, 0))
        .message(newMessage)
        .save();
    }
    return visitorContext;
  }

  private void executeChecks(PythonVisitorContext visitorContext, Collection<PythonCheck> checks, InputFile.Type fileType, PythonInputFile inputFile) {
    Collection<PythonSubscriptionCheck> subscriptionChecks = new ArrayList<>();
    for (PythonCheck check : checks) {
      if (isCheckNotApplicable(check, fileType)
        || checksExecutedWithoutParsingByFiles.getOrDefault(inputFile, Collections.emptySet()).contains(check.getClass())) {
        continue;
      }
      if (check instanceof PythonSubscriptionCheck pythonSubscriptionCheck) {
        subscriptionChecks.add(pythonSubscriptionCheck);
      } else {
        check.scanFile(visitorContext);
      }
    }
    SubscriptionVisitor.analyze(subscriptionChecks, visitorContext);
  }

  private void executeOtherChecks(PythonInputFile inputFile, PythonVisitorContext visitorContext, InputFile.Type fileType) {
    checks.noSonarPythonChecks().forEach((repositoryKey, repositoryChecks) -> {
        var lock = repositoryLocks.computeIfAbsent(repositoryKey, k -> new Object());
        synchronized (lock) {
          executeChecks(visitorContext, repositoryChecks, fileType, inputFile);
        }
      }
    );
  }

  private void searchForDataBricks(PythonVisitorContext visitorContext) {
    var hasDatabricks = visitorContext.pythonFile().content().lines().anyMatch(
      line -> DATABRICKS_MAGIC_COMMAND_PATTERN.matcher(line).matches());
    foundDatabricks.compareAndSet(false, hasDatabricks);
  }

  private static PythonTreeMaker getTreeMaker(PythonInputFile inputFile) {
    return Python.KEY.equals(inputFile.wrappedFile().language()) ? new PythonTreeMaker() :
      new IPythonTreeMaker(getOffsetLocations(inputFile));
  }

  private static Map<Integer, IPythonLocation> getOffsetLocations(PythonInputFile inputFile) {
    if (inputFile.kind() == PythonInputFile.Kind.IPYTHON) {
      return ((GeneratedIPythonFile) inputFile).locationMap();
    }
    return Map.of();
  }

  @Override
  public boolean scanFileWithoutParsing(PythonInputFile inputFile) {
    InputFile.Type fileType = inputFile.wrappedFile().type();
    boolean result = true;
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile.wrappedFile());
    PythonInputFileContext inputFileContext = new PythonInputFileContext(
      pythonFile,
      context.fileSystem().workDir(),
      indexer.cacheContext(),
      context.runtime().getProduct(),
      indexer.projectLevelSymbolTable()
    );

    result = scanFileWithoutParsingSonarPython(inputFile, fileType, inputFileContext, result);

    var otherChecks = checks.noSonarPythonChecks();
    for (var entry : otherChecks.entrySet()) {
      var repositoryKey = entry.getKey();
      var repositoryChecks = entry.getValue();
      var lock = repositoryLocks.computeIfAbsent(repositoryKey, k -> new Object());
      synchronized (lock) {
        for (var check : repositoryChecks) {
          result = scanFileWithoutParsingNotSonarPython(inputFile, check, fileType, result, inputFileContext);
        }
      }
    }

    result &= architectureCallback.scanWithoutParsing(inputFileContext);
    if (!result) {
      // If scan without parsing is not successful, measures will be pushed during regular scan.
      // We must avoid pushing measures twice due to the risk of duplicate cache key error.
      return false;
    }
    return restoreAndPushMeasuresIfApplicable(inputFile);
  }

  private boolean scanFileWithoutParsingNotSonarPython(PythonInputFile inputFile, PythonCheck check, InputFile.Type fileType, boolean result,
    PythonInputFileContext inputFileContext) {
    if (isCheckNotApplicable(check, fileType)) {
      return result;
    }
    if (!indexer.canBeFullyScannedWithoutParsing(inputFile)) {
      result = false;
      return result;
    }
    if (check.scanWithoutParsing(inputFileContext)) {
      var executedChecks = checksExecutedWithoutParsingByFiles.getOrDefault(inputFile, new HashSet<>());
      executedChecks.add(check.getClass());
      checksExecutedWithoutParsingByFiles.putIfAbsent(inputFile, executedChecks);
    } else {
      result = false;
    }
    return result;
  }

  private boolean scanFileWithoutParsingSonarPython(PythonInputFile inputFile, InputFile.Type fileType, PythonInputFileContext inputFileContext, boolean result) {
    var ourChecks = checks.sonarPythonChecks();
    for (var check : ourChecks) {
      if (isCheckNotApplicable(check, fileType)) {
        continue;
      }
      if (check.scanWithoutParsing(inputFileContext)) {
        var executedChecks = checksExecutedWithoutParsingByFiles.getOrDefault(inputFile, new HashSet<>());
        executedChecks.add(check.getClass());
        checksExecutedWithoutParsingByFiles.putIfAbsent(inputFile, executedChecks);
      } else {
        result = false;
      }
    }
    return result;
  }


  @Override
  public void endOfAnalysis() {
    indexer.postAnalysis(context);
    checks.sonarPythonEndOfAnalyses().forEach(c -> c.endOfAnalysis(indexer.cacheContext()));
    checks.noSonarPythonEndOfAnalyses().forEach(
      (repositoryKey, endOfAnalyses) -> {
        var lock = repositoryLocks.computeIfAbsent(repositoryKey, k -> new Object());
        synchronized (lock) {
          endOfAnalyses.forEach(c -> c.endOfAnalysis(indexer.cacheContext()));
        }
      }
    );
  }

  boolean isCheckNotApplicable(PythonCheck pythonCheck, InputFile.Type fileType) {
    return fileType != InputFile.Type.MAIN && pythonCheck.scope() != PythonCheck.CheckScope.ALL;
  }

  // visible for testing
  static File getWorkingDirectory(SensorContext context) {
    return isInSonarLint(context) ? null : context.fileSystem().workDir();
  }

  private static boolean isInSonarLint(SensorContext context) {
    return context.runtime().getProduct().equals(SonarProduct.SONARLINT);
  }

  @Override
  protected void processException(Exception e, PythonInputFile file) {
    LOG.warn("Unable to analyze file: " + file, e);
  }

  @Override
  public boolean canBeScannedWithoutParsing(PythonInputFile inputFile) {
    return this.indexer.canBePartiallyScannedWithoutParsing(inputFile);
  }

  @Override
  protected void reportStatistics(int numSkippedFiles, int numTotalFiles) {
    LOG.info("The Python analyzer was able to leverage cached data from previous analyses for {} out of {} files. These files were not " +
             "parsed.",
      numSkippedFiles, numTotalFiles);
  }

  private synchronized void saveIssues(PythonInputFile inputFile, List<PreciseIssue> issues) {
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
            secondaryLocationsFlow.addFirst(newLocation(new PythonInputFileImpl(issueLocationFile), newIssue, secondaryLocation));
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

      handleQuickFixes(inputFile.wrappedFile(), ruleKey, newIssue, preciseIssue);

      newIssue.save();
    }
  }

  @CheckForNull
  private InputFile component(String fileId, SensorContext sensorContext) {
    var predicate = sensorContext.fileSystem().predicates().is(new File(fileId));
    InputFile inputFile = Optional.ofNullable(sensorContext.fileSystem().inputFile(predicate))
      .orElseGet(() -> indexer.getFileWithId(fileId));
    if (inputFile == null) {
      LOG.debug("Failed to find InputFile for {}", fileId);
    }
    return inputFile;
  }

  private static NewIssueLocation newLocation(PythonInputFile inputFile, NewIssue issue, IssueLocation location) {
    NewIssueLocation newLocation = issue.newLocation()
      .on(inputFile.wrappedFile());
    if (location.startLine() != IssueLocation.UNDEFINED_LINE) {
      TextRange range;
      if (location.startLineOffset() == IssueLocation.UNDEFINED_OFFSET) {
        range = inputFile.wrappedFile().selectLine(location.startLine());
      } else {
        range = inputFile.wrappedFile().newRange(location.startLine(), location.startLineOffset(), location.endLine(),
          location.endLineOffset());
      }
      newLocation.at(range);
    }

    String message = location.message();
    if (message != null) {
      newLocation.message(message);
    }
    return newLocation;
  }

  private synchronized void saveMeasures(PythonInputFile inputFile, PythonVisitorContext visitorContext) {
    FileMetrics fileMetrics = new FileMetrics(visitorContext, isNotebook(inputFile));
    FileLinesVisitor fileLinesVisitor = fileMetrics.fileLinesVisitor();

    noSonarFilter.noSonarInFile(inputFile.wrappedFile(), fileLinesVisitor.getLinesWithNoSonar());

    if (!isInSonarLint(context)) {
      if (inputFile.kind() == PythonInputFile.Kind.PYTHON) {
        cpdAnalyzer.pushCpdTokens(inputFile.wrappedFile(), visitorContext);
      }

      Set<Integer> linesOfCode = fileLinesVisitor.getLinesOfCode();
      saveMetricOnFile(inputFile, CoreMetrics.NCLOC, linesOfCode.size());
      saveMetricOnFile(inputFile, CoreMetrics.STATEMENTS, fileMetrics.numberOfStatements());
      saveMetricOnFile(inputFile, CoreMetrics.FUNCTIONS, fileMetrics.numberOfFunctions());
      saveMetricOnFile(inputFile, CoreMetrics.CLASSES, fileMetrics.numberOfClasses());
      saveMetricOnFile(inputFile, CoreMetrics.COMPLEXITY, fileMetrics.complexity());
      saveMetricOnFile(inputFile, CoreMetrics.COGNITIVE_COMPLEXITY, fileMetrics.cognitiveComplexity());
      saveMetricOnFile(inputFile, CoreMetrics.COMMENT_LINES, fileLinesVisitor.getCommentLineCount());

      FileLinesContext fileLinesContext = fileLinesContextFactory.createFor(inputFile.wrappedFile());
      if (inputFile.kind() == PythonInputFile.Kind.PYTHON) {
        for (int line : linesOfCode) {
          fileLinesContext.setIntValue(CoreMetrics.NCLOC_DATA_KEY, line, 1);
        }
      }
      for (int line : fileLinesVisitor.getExecutableLines()) {
        fileLinesContext.setIntValue(CoreMetrics.EXECUTABLE_LINES_DATA_KEY, line, 1);
      }
      fileLinesContext.save();
    }
  }

  static boolean isNotebook(PythonInputFile inputFile) {
    return inputFile.kind() == PythonInputFile.Kind.IPYTHON;
  }

  private synchronized boolean restoreAndPushMeasuresIfApplicable(PythonInputFile inputFile) {
    if (inputFile.wrappedFile().type() == InputFile.Type.TEST) {
      return true;
    }

    return cpdAnalyzer.pushCachedCpdTokens(inputFile.wrappedFile(), indexer.cacheContext());
  }

  private void saveMetricOnFile(PythonInputFile inputFile, Metric<Integer> metric, Integer value) {
    context.<Integer>newMeasure()
      .withValue(value)
      .forMetric(metric)
      .on(inputFile.wrappedFile())
      .save();
  }

  private void handleQuickFixes(InputFile inputFile, RuleKey ruleKey, NewIssue newIssue, PreciseIssue preciseIssue) {
    if (isInSonarLint(context)) {
      List<PythonQuickFix> quickFixes = preciseIssue.quickFixes();
      addQuickFixes(inputFile, ruleKey, quickFixes, newIssue);
    }
  }

  private static void addQuickFixes(InputFile inputFile, RuleKey ruleKey, Iterable<PythonQuickFix> quickFixes, NewIssue sonarLintIssue) {
    try {
      for (PythonQuickFix quickFix : quickFixes) {
        var newQuickFix = sonarLintIssue.newQuickFix()
          .message(quickFix.getDescription());

        var edit = newQuickFix.newInputFileEdit().on(inputFile);

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
    return file.newRange(pythonTextEdit.startLine(), pythonTextEdit.startLineOffset(), pythonTextEdit.endLine(),
      pythonTextEdit.endLineOffset());
  }

  public int getRecognitionErrorCount() {
    return recognitionErrorCount.get();
  }

  public boolean getFoundDatabricks() {
    return foundDatabricks.get();
  }

  private static Integer getNumberOfThreads(SensorContext context) {
    return context.config().getInt(THREADS_PROPERTY_NAME)
      .orElse(1);
  }
}
