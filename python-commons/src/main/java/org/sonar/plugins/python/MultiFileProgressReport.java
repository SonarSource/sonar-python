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


import com.google.common.annotations.VisibleForTesting;
import java.time.Instant;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MultiFileProgressReport implements Runnable {

  private static final Logger LOG = LoggerFactory.getLogger(MultiFileProgressReport.class);
  private static final int MAX_NUMBER_OF_FILES_TO_DISPLAY = 3;
  private static final int DEFAULT_PROGRESS_UPDATE_PERIOD_MILLIS = 10000;

  // data structure is chosen because of the preservation of insertion order. This allows us to display the longest running files first.
  private final Collection<String> currentFileNames = new ConcurrentLinkedDeque<>();
  private long size;
  private long numberOfFinishedFiles;
  private final Thread thread;
  private final long progressUpdatePeriod;
  private boolean success;
  private Instant startInstant;

  private BiConsumer<String, Boolean> logFunction = MultiFileProgressReport::defaultLogFunction;

  /**
   * The report loop can not rely only on Thread.interrupted() to end, according to
   * interrupted() javadoc, a thread interruption can be ignored because a thread was
   * not alive at the time of the interrupt. This could happen if stop() is being called
   * before ProgressReport's thread becomes alive.
   * So this boolean flag ensures that ProgressReport never enter an infinite loop when
   * Thread.interrupted() failed to be set to true.
   */
  private final AtomicBoolean interrupted = new AtomicBoolean();

  @VisibleForTesting
  MultiFileProgressReport(long progressUpdatePeriod, BiConsumer<String, Boolean> logFunction) {
    this(progressUpdatePeriod, "default");
    this.logFunction = logFunction;
  }

  public MultiFileProgressReport(String stepName) {
    this(DEFAULT_PROGRESS_UPDATE_PERIOD_MILLIS, stepName);
  }

  public MultiFileProgressReport(long progressUpdatePeriod) {
    this(progressUpdatePeriod, "default");
  }

  public MultiFileProgressReport(long progressUpdatePeriod, String stepName) {
    this.progressUpdatePeriod = progressUpdatePeriod;
    interrupted.set(false);
    thread = new Thread(this);
    thread.setName(stepName);
    thread.setDaemon(true);
    thread.setUncaughtExceptionHandler((thread, throwable) -> LOG.debug("Uncaught exception in the progress report thread: {}", throwable.getClass().getCanonicalName()));
  }

  @Override
  public void run() {
    startInstant = Instant.now();
    log(size + " source " + pluralizeFile(size) + " to be analyzed", false);
    while (!(interrupted.get() || Thread.currentThread().isInterrupted())) {
      try {
        Thread.sleep(progressUpdatePeriod);
        logCurrentProgress();
      } catch (InterruptedException e) {
        interrupted.set(true);
        thread.interrupt();
        break;
      }
    }
    if (success) {
      log(size + "/" + size + " source " + pluralizeFile(size) + " " + pluralizeHas(size) + " been analyzed", false);
    }
  }

  private static String pluralizeFile(long count) {
    if (count == 1L) {
      return "file";
    }
    return "files";
  }

  private static String pluralizeHas(long count) {
    if (count == 1L) {
      return "has";
    }
    return "have";
  }

  public synchronized void start(int size) {
    this.size = size;
    thread.start();
  }

  public void startAnalysisFor(String fileName) {
    currentFileNames.add(fileName);
  }

  public synchronized void finishAnalysisFor(String fileName) {
    if (!currentFileNames.remove(fileName)) {
      log("Couldn't finish progress report of file \"%s\", as it was not in the list of files being analyzed".formatted(fileName), true);
      return;
    }
    if (numberOfFinishedFiles < size) {
      numberOfFinishedFiles++;
    } else {
      log("Reported finished analysis on more files than expected", true);
    }
  }

  public synchronized void stop() {
    interrupted.set(true);
    success = true;
    thread.interrupt();
    join();
    logFinishTime();
  }

  public synchronized void cancel() {
    interrupted.set(true);
    thread.interrupt();
    join();
    logFinishTime();
  }

  private void join() {
    try {
      thread.join();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  private void logCurrentProgress() {
    var sb = new StringBuilder();
    Collection<String> currentFileNamesCopy;
    synchronized (this) {
      currentFileNamesCopy = new LinkedHashSet<>(currentFileNames);
    }
    int numberOfFiles = currentFileNamesCopy.size();
    sb.append(numberOfFinishedFiles)
      .append("/")
      .append(size)
      .append(" files analyzed, current ")
      .append(pluralizeFile(numberOfFiles))
      .append(": ");

    boolean debugEnabled = LOG.isDebugEnabled();
    if (numberOfFiles == 0) {
      sb.append("none");
    } else {
      int numberOfFilesToDisplay = debugEnabled ? numberOfFiles : Math.min(numberOfFiles, MAX_NUMBER_OF_FILES_TO_DISPLAY);
      var fileNamesToDisplay = currentFileNamesCopy.stream()
        .limit(numberOfFilesToDisplay)
        .collect(Collectors.joining(", "));
      sb.append(fileNamesToDisplay);
      if (numberOfFiles > numberOfFilesToDisplay) {
        sb.append(", ...");
      }
    }

    log(sb.toString(), debugEnabled);
  }

  private void logFinishTime() {
    log("Finished step " + thread.getName() + " in " + (Instant.now().toEpochMilli() - startInstant.toEpochMilli()) + "ms", false);
  }

  private void log(String message, boolean debug) {
    logFunction.accept(message, debug);
  }

  private static void defaultLogFunction(String message, boolean debug) {
    if (debug) {
      LOG.debug(message);
    } else {
      LOG.info(message);
    }
  }
}
