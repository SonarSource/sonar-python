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
package org.sonar.plugins.python.indexer;

import java.util.List;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;

public class SonarQubePythonIndexer extends PythonIndexer {

  private final List<InputFile> files;
  private static final Logger LOG = Loggers.get(SonarQubePythonIndexer.class);

  public SonarQubePythonIndexer(List<InputFile> files) {
    this.files = files;
  }

  @Override
  public void buildOnce(SensorContext context) {
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    LOG.debug("Input files for indexing: " + files);
    // computes "globalSymbolsByModuleName"
    long startTime = System.currentTimeMillis();
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
    long stopTime = System.currentTimeMillis() - startTime;
    LOG.debug("Time to build the project level symbol table: " + stopTime + "ms");
  }
}
