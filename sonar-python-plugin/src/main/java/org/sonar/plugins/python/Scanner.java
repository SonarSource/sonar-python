/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonarsource.analyzer.commons.ProgressReport;

public abstract class Scanner {
  private static final Logger LOG = LoggerFactory.getLogger(Scanner.class);
  private static final String FAIL_FAST_PROPERTY_NAME = "sonar.internal.analysis.failFast";
  protected final SensorContext context;

  protected Scanner(SensorContext context) {
    this.context = context;
  }

  public void execute(List<InputFile> files, SensorContext context) {
    ProgressReport progressReport = new ProgressReport(this.name() + " progress", TimeUnit.SECONDS.toMillis(10));
    LOG.info("Starting " + this.name());
    List<String> filenames = files.stream().map(InputFile::toString).collect(Collectors.toList());

    int numScannedWithoutParsing = 0;
    progressReport.start(filenames);
    for (InputFile file : files) {
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
        if (context.config().getBoolean(FAIL_FAST_PROPERTY_NAME).orElse(false)) {
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

  protected abstract void scanFile(InputFile file) throws IOException;

  protected boolean scanFileWithoutParsing(InputFile file) throws IOException {
    return false;
  }

  protected void endOfAnalysis() {
    // no op
  }

  protected abstract void processException(Exception e, InputFile file);

  protected void reportStatistics(int numSkippedFiles, int numTotalFiles) {
    // Intentionally empty. Subclasses can override this method to output logs containing some logs after the execution of the scanner.
  }

  public boolean canBeScannedWithoutParsing(InputFile inputFile) {
    return false;
  }
}
