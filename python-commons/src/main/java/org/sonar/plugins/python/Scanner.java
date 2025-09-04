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

import com.sonar.sslr.api.RecognitionException;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;

public abstract class Scanner {
  private static final Logger LOG = LoggerFactory.getLogger(Scanner.class);
  private static final String FAIL_FAST_PROPERTY_NAME = "sonar.internal.analysis.failFast";
  public static final String THREADS_PROPERTY_NAME = "sonar.python.analysis.threads";
  protected final SensorContext context;

  protected Scanner(SensorContext context) {
    this.context = context;
  }

  public void execute(List<PythonInputFile> files, SensorContext context) {
    var progressReport = new MultiFileProgressReport(name());
    String name = this.name();
    LOG.info("Starting {}", name);
    List<String> filenames = getFilesStream(files).map(PythonInputFile::wrappedFile).map(InputFile::toString).toList();

    var numScannedWithoutParsing = new AtomicInteger();
    progressReport.start(filenames.size());
    processFiles(files, context, progressReport, numScannedWithoutParsing);
    endOfAnalysis();
    progressReport.stop();
    this.reportStatistics(numScannedWithoutParsing.get(), files.size());
  }

  protected void processFiles(List<PythonInputFile> files, SensorContext context, MultiFileProgressReport progressReport, AtomicInteger numScannedWithoutParsing) {
    var numberOfThreads = getNumberOfThreads(context);
    logStart(numberOfThreads);
    if (numberOfThreads == 1) {
      getFilesStream(files).forEach(file -> processFile(context, file, progressReport, numScannedWithoutParsing));
      return;
    }
    var executor = Executors.newWorkStealingPool(numberOfThreads);
    try {
      var allTasks = CompletableFuture.allOf(
        files.stream()
          .map(file -> CompletableFuture.runAsync(() -> processFile(context, file, progressReport, numScannedWithoutParsing), executor))
          .toArray(CompletableFuture[]::new)
      );
      allTasks.join();
    } catch (CompletionException e) {
      var cause = e.getCause();
      if (cause instanceof RuntimeException runtimeException) {
        throw runtimeException;
      } else if (cause instanceof Error error) {
        throw error;
      } else {
        throw e;
      }
    } finally {
      executor.shutdown();
    }
  }

  protected abstract void logStart(int numThreads);

  protected Stream<PythonInputFile> getFilesStream(List<PythonInputFile> files) {
    return files.stream();
  }

  private void processFile(SensorContext context, PythonInputFile file, MultiFileProgressReport progressReport, AtomicInteger numScannedWithoutParsing) {
    if (context.isCancelled()) {
      progressReport.cancel();
      return;
    }
    var filename = file.wrappedFile().filename();
    try {
      progressReport.startAnalysisFor(filename);
      boolean successfullyScannedWithoutParsing = false;
      if (canBeScannedWithoutParsing(file)) {
        successfullyScannedWithoutParsing = this.scanFileWithoutParsing(file);
      }
      if (!successfullyScannedWithoutParsing) {
        this.scanFile(file);
      } else {
        numScannedWithoutParsing.incrementAndGet();
      }
    } catch (Exception e) {
      this.processException(e, file);
      if (context.config().getBoolean(FAIL_FAST_PROPERTY_NAME).orElse(false) && !isParseErrorOnTestFile(file, e)) {
        throw new IllegalStateException("Exception when analyzing " + file, e);
      }
    } finally {
      progressReport.finishAnalysisFor(filename);
    }
  }

  protected abstract String name();

  protected abstract void scanFile(PythonInputFile file) throws IOException;

  protected boolean scanFileWithoutParsing(PythonInputFile file) throws IOException {
    return false;
  }

  protected void endOfAnalysis() {
    // no op
  }

  protected abstract void processException(Exception e, PythonInputFile file);

  protected void reportStatistics(int numSkippedFiles, int numTotalFiles) {
    // Intentionally empty. Subclasses can override this method to output logs containing some logs after the execution of the scanner.
  }

  public boolean canBeScannedWithoutParsing(PythonInputFile inputFile) {
    return false;
  }

  private static boolean isParseErrorOnTestFile(PythonInputFile file, Exception e) {
    // As test files may contain invalid syntax on purpose, we avoid failing the analysis when encountering parse errors on them
    return e instanceof RecognitionException && file.wrappedFile().type() == InputFile.Type.TEST;
  }

  protected int getNumberOfThreads(SensorContext context) {
    int minNumOfThreads = 1;
    int maxNumOfThreads = 6;
    int availableProcessors = (int) Math.round(Runtime.getRuntime().availableProcessors() * 0.9);

    // Disabling parallelization if threads property is not setup properly
    return context.config()
      .getInt(THREADS_PROPERTY_NAME)
      .map(threads -> threads < 1 ? 1 : threads)
      .orElse(Math.max(minNumOfThreads, Math.min(availableProcessors, maxNumOfThreads)));
  }

}
