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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonarsource.api.sonarlint.SonarLintSide;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileEvent;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileListener;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileSystem;

@SonarLintSide()
public class SonarLintPythonIndexer extends PythonIndexer implements ModuleFileListener {

  private final ModuleFileSystem moduleFileSystem;

  public SonarLintPythonIndexer(ModuleFileSystem moduleFileSystem) {
    this.moduleFileSystem = moduleFileSystem;
  }

  @Override
  public void buildOnce(SensorContext context) {
    this.projectBaseDir = context.fileSystem().baseDir();
    List<InputFile> files = getInputFiles(moduleFileSystem);
    LOG.debug("Input files for indexing: " + files);
    // computes "globalSymbolsByModuleName"
    long startTime = System.currentTimeMillis();
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
    long stopTime = System.currentTimeMillis() - startTime;
    LOG.debug("Time to build the project level symbol table: " + stopTime + "ms");
  }

  private List<InputFile> getInputFiles(ModuleFileSystem moduleFileSystem) {
    List<InputFile> files = new ArrayList<>();
    moduleFileSystem.files(Python.KEY, InputFile.Type.MAIN, files::add);
    return Collections.unmodifiableList(files);
  }

  @Override
  public void process(ModuleFileEvent moduleFileEvent) {
    InputFile target = moduleFileEvent.getTarget();
    ModuleFileEvent.Type type = moduleFileEvent.getType();
    if (type.equals(ModuleFileEvent.Type.DELETED) || type.equals(ModuleFileEvent.Type.MODIFIED)) {
      removeFile(target);
    }
    if (type.equals(ModuleFileEvent.Type.CREATED) || type.equals(ModuleFileEvent.Type.MODIFIED)) {
      try {
        addFile(target);
      } catch (IOException e) {
        LOG.warn("Failed to load file \"{}\" to the project symbol table", target.filename());
      }
    }
  }
}
