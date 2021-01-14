/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonarsource.analyzer.commons.ProgressReport;

abstract class Scanner {
  private static final Logger LOG = Loggers.get(Scanner.class);
  private static final String FAIL_FAST_PROPERTY_NAME = "sonar.internal.analysis.failFast";
  protected final SensorContext context;

  Scanner(SensorContext context) {
    this.context = context;
  }

  void execute(List<InputFile> files, SensorContext context) {
    ProgressReport progressReport = new ProgressReport(this.name() + " progress", TimeUnit.SECONDS.toMillis(10));
    LOG.info("Starting " + this.name());
    List<String> filenames = files.stream().map(InputFile::toString).collect(Collectors.toList());
    progressReport.start(filenames);
    for (InputFile file : files) {
      if (context.isCancelled()) {
        progressReport.cancel();
        return;
      }
      try {
        this.scanFile(file);
      } catch (Exception e) {
        this.processException(e, file);
        if (context.config().getBoolean(FAIL_FAST_PROPERTY_NAME).orElse(false)) {
          throw new IllegalStateException("Exception when analyzing " + file, e);
        }
      } finally {
        progressReport.nextFile();
      }
    }

    progressReport.stop();
  }

  abstract String name();

  abstract void scanFile(InputFile file) throws IOException;

  abstract void processException(Exception e, InputFile file);
}
