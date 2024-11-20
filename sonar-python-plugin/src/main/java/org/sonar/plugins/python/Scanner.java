/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonarsource.analyzer.commons.ProgressReport;

public abstract class Scanner {
  private static final Logger LOG = LoggerFactory.getLogger(Scanner.class);
  private static final String FAIL_FAST_PROPERTY_NAME = "sonar.internal.analysis.failFast";
  protected final SensorContext context;

  protected Scanner(SensorContext context) {
    this.context = context;
  }

  public void execute(List<PythonInputFile> files, SensorContext context) {
    ProgressReport progressReport = new ProgressReport(this.name() + " progress", TimeUnit.SECONDS.toMillis(10));
    String name = this.name();
    LOG.info("Starting {}", name);
    List<String> filenames = files.stream().map(PythonInputFile::wrappedFile).map(InputFile::toString).toList();

    int numScannedWithoutParsing = 0;
    progressReport.start(filenames);
    for (PythonInputFile file : files) {
      if (context.isCancelled()) {
        progressReport.cancel();
        return;
      }
      try {
        boolean successfullyScannedWithoutParsing = false;
        if (canBeScannedWithoutParsing(file)) {
          successfullyScannedWithoutParsing = this.scanFileWithoutParsing(file);
        }
        if (!successfullyScannedWithoutParsing) {
          this.scanFile(file);
        } else {
          ++numScannedWithoutParsing;
        }
      } catch (Exception e) {
        this.processException(e, file);
        if (context.config().getBoolean(FAIL_FAST_PROPERTY_NAME).orElse(false) && !isParseErrorOnTestFile(file, e)) {
          throw new IllegalStateException("Exception when analyzing " + file, e);
        }
      } finally {
        progressReport.nextFile();
      }
    }
    endOfAnalysis();
    progressReport.stop();
    this.reportStatistics(numScannedWithoutParsing, files.size());
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
}
