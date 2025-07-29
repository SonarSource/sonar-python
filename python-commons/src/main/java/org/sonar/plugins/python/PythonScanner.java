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
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.SonarProduct;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonFileConsumer;
import org.sonar.plugins.python.api.PythonInputFileContext;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.internal.EndOfAnalysis;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.cpd.PythonCpdAnalyzer;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.nosonar.NoSonarLineInfoCollector;
import org.sonar.python.IPythonLocation;
import org.sonar.python.SubscriptionVisitor;
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
  private final PythonCpdAnalyzer cpdAnalyzer;
  private final PythonIndexer indexer;
  private final Map<PythonInputFile, Set<Class<? extends PythonCheck>>> checksExecutedWithoutParsingByFiles;
  private final AtomicInteger recognitionErrorCount;
  private final AtomicBoolean foundDatabricks;
  private final PythonFileConsumer architectureCallback;
  private final Map<String, Lock> repositoryLocks;
  private final NewSymbolsCollector newSymbolsCollector;
  private final PythonHighlighter pythonHighlighter;
  private final IssuesRepository issuesRepository;
  private final MeasuresRepository measuresRepository;
  private final NoSonarLineInfoCollector noSonarLineInfoCollector;
  private final Lock lock;

  public PythonScanner(
    SensorContext context, PythonChecks checks, FileLinesContextFactory fileLinesContextFactory, NoSonarFilter noSonarFilter,
    Supplier<PythonParser> parserSupplier, PythonIndexer indexer, PythonFileConsumer architectureCallback, NoSonarLineInfoCollector noSonarLineInfoCollector) {
    super(context);
    this.checks = checks;
    this.parserSupplier = parserSupplier;
    this.indexer = indexer;
    this.noSonarLineInfoCollector = noSonarLineInfoCollector;
    this.indexer.buildOnce(context);
    this.architectureCallback = architectureCallback;
    this.checksExecutedWithoutParsingByFiles = new ConcurrentHashMap<>();
    this.recognitionErrorCount = new AtomicInteger(0);
    this.foundDatabricks = new AtomicBoolean(false);
    this.repositoryLocks = new ConcurrentHashMap<>();
    this.lock = new ReentrantLock();
    this.cpdAnalyzer = new PythonCpdAnalyzer(context, lock);
    this.newSymbolsCollector = new NewSymbolsCollector(lock);
    this.pythonHighlighter = new PythonHighlighter(lock);
    this.issuesRepository = new IssuesRepository(context, checks, indexer, isInSonarLint(context), lock);
    this.measuresRepository = new MeasuresRepository(context, noSonarFilter, fileLinesContextFactory, isInSonarLint(context), noSonarLineInfoCollector, lock);
  }

  @Override
  protected String name() {
    return "rules execution";
  }

  @Override
  protected void logStart(int numThreads) {
    LOG.debug("Scanning files in {} threads", numThreads);
  }

  @Override
  protected void scanFile(PythonInputFile inputFile) throws IOException {
    var pythonFile = SonarQubePythonFile.create(inputFile);
    InputFile.Type fileType = inputFile.wrappedFile().type();
    PythonVisitorContext visitorContext = createVisitorContext(inputFile, pythonFile, fileType);

    executeChecks(visitorContext, checks.sonarPythonChecks(), fileType, inputFile);
    executeOtherChecks(inputFile, visitorContext, fileType);


    runLockedByRepository(ARCHITECTURE_CALLBACK_LOCK_KEY, () -> architectureCallback.scanFile(visitorContext));


    noSonarLineInfoCollector.collect(pythonFile.key(), visitorContext.rootTree());

    if (fileType == InputFile.Type.MAIN && visitorContext.rootTree() != null) {
      pushTokens(inputFile, visitorContext);
      measuresRepository.save(inputFile, visitorContext);
    }

    var issues = visitorContext.getIssues();
    issuesRepository.save(inputFile, issues);

    if (visitorContext.rootTree() != null && !isInSonarLint(context)) {
      newSymbolsCollector.collect(context.newSymbolTable().onFile(inputFile.wrappedFile()), visitorContext.rootTree());
      pythonHighlighter.highlight(context, visitorContext, inputFile);
    }

    searchForDataBricks(visitorContext);
  }

  private PythonVisitorContext createVisitorContext(PythonInputFile inputFile, PythonFile pythonFile, InputFile.Type fileType) throws IOException {
    PythonVisitorContext visitorContext;
    try {
      AstNode astNode = parserSupplier.get().parse(inputFile.contents());
      PythonTreeMaker treeMaker = getTreeMaker(inputFile);
      FileInput parse = treeMaker.fileInput(astNode);
      visitorContext = new PythonVisitorContext.Builder(parse, pythonFile)
        .projectConfiguration(indexer.projectConfig())
        .workingDirectory(getWorkingDirectory(context))
        .packageName(indexer.packageName(inputFile))
        .projectLevelSymbolTable(indexer.projectLevelSymbolTable())
        .cacheContext(indexer.cacheContext())
        .sonarProduct(context.runtime().getProduct())
        .build();

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

  private void pushTokens(PythonInputFile inputFile, PythonVisitorContext visitorContext) {
    if (!isInSonarLint(context) && inputFile.kind() == PythonInputFile.Kind.PYTHON) {
      cpdAnalyzer.pushCpdTokens(inputFile.wrappedFile(), visitorContext);
    }
  }

  private void executeChecks(PythonVisitorContext visitorContext, Collection<PythonCheck> checks, InputFile.Type fileType,
    PythonInputFile inputFile) {
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
    checks.noSonarPythonChecks()
      .forEach((repositoryKey, repositoryChecks) -> runLockedByRepository(repositoryKey, () -> executeChecks(visitorContext, repositoryChecks, fileType, inputFile)));
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

    var atomicResult = new AtomicBoolean(result);
    var otherChecks = checks.noSonarPythonChecks();
    otherChecks.forEach((repositoryKey, repositoryChecks) -> runLockedByRepository(repositoryKey, () -> {
      for (var check : repositoryChecks) {
        var scanResult = scanFileWithoutParsingNotSonarPython(inputFile, check, fileType, atomicResult.get(), inputFileContext);
        atomicResult.set(scanResult);
      }
    }));
    result = atomicResult.get();

    result &= architectureCallback.scanWithoutParsing(inputFileContext);
    if (!result) {
      // If scan without parsing is not successful, measures will be pushed during regular scan.
      // We must avoid pushing measures twice due to the risk of duplicate cache key error.
      return false;
    }
    return restoreAndPushMeasuresIfApplicable(inputFile);
  }

  private boolean scanFileWithoutParsingNotSonarPython(PythonInputFile inputFile, PythonCheck check, InputFile.Type fileType,
    boolean result,
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

  private boolean scanFileWithoutParsingSonarPython(PythonInputFile inputFile, InputFile.Type fileType,
    PythonInputFileContext inputFileContext, boolean result) {
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
    checks.noSonarPythonEndOfAnalyses().forEach(this::endOfAnalysisForRepository);
  }

  private void endOfAnalysisForRepository(String repositoryKey, List<EndOfAnalysis> endOfAnalyses) {
    runLockedByRepository(repositoryKey, () -> endOfAnalyses.forEach(c -> c.endOfAnalysis(indexer.cacheContext())));
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
    LOG.info("""
             The Python analyzer was able to leverage cached data from previous analyses for {} out of {} files. These files were not \
             parsed.""",
      numSkippedFiles, numTotalFiles);
  }

  private boolean restoreAndPushMeasuresIfApplicable(PythonInputFile inputFile) {
    if (inputFile.wrappedFile().type() == InputFile.Type.TEST) {
      return true;
    }

    return cpdAnalyzer.pushCachedCpdTokens(inputFile.wrappedFile(), indexer.cacheContext());
  }

  public int getRecognitionErrorCount() {
    return recognitionErrorCount.get();
  }

  public boolean getFoundDatabricks() {
    return foundDatabricks.get();
  }

  @Override
  protected int getNumberOfThreads(SensorContext context) {
    return context.config().getInt(THREADS_PROPERTY_NAME)
      .orElse(1);
  }

  private void runLockedByRepository(String repositoryKey, Runnable runnable) {
    var repositoryLock = repositoryLocks.computeIfAbsent(repositoryKey, k -> new ReentrantLock());
    try {
      repositoryLock.lock();
      runnable.run();
    } finally {
      repositoryLock.unlock();
    }
  }
}
