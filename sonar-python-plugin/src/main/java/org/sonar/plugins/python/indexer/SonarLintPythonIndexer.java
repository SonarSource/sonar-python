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
package org.sonar.plugins.python.indexer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.Python;
import org.sonarsource.api.sonarlint.SonarLintSide;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileEvent;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileListener;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileSystem;

@SonarLintSide(lifespan = "MODULE")
public class SonarLintPythonIndexer extends PythonIndexer implements ModuleFileListener {

  private final ModuleFileSystem moduleFileSystem;

  private final Map<String, InputFile> indexedFiles = new HashMap<>();
  private static final Logger LOG = LoggerFactory.getLogger(SonarLintPythonIndexer.class);
  private boolean shouldBuildProjectSymbolTable = true;
  private static final long DEFAULT_MAX_LINES_FOR_INDEXING = 300_000;
  private static final String MAX_LINES_PROPERTY = "sonar.python.sonarlint.indexing.maxlines";

  public SonarLintPythonIndexer(ModuleFileSystem moduleFileSystem) {
    this.moduleFileSystem = moduleFileSystem;
  }

  @Override
  public void buildOnce(SensorContext context) {
    if (!shouldBuildProjectSymbolTable) {
      return;
    }
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    shouldBuildProjectSymbolTable = false;
    List<InputFile> files = getInputFiles(moduleFileSystem);
    long nLines = files.stream().map(InputFile::lines).mapToLong(Integer::longValue).sum();
    long maxLinesForIndexing = context.config().getLong(MAX_LINES_PROPERTY).orElse(DEFAULT_MAX_LINES_FOR_INDEXING);
    if (nLines > maxLinesForIndexing) {
      // Avoid performance issues for large projects
      LOG.debug("Project symbol table deactivated due to project size (total number of lines is {}, maximum for indexing is {})", nLines, maxLinesForIndexing);
      LOG.debug("Update \"sonar.python.sonarlint.indexing.maxlines\" to set a different limit.");
      return;
    }
    LOG.debug("Input files for indexing: {}", files);
    // computes "globalSymbolsByModuleName"
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
  }

  @Override
  public InputFile getFileWithId(String fileId) {
    String compare = fileId.replace("\\", "/");
    return indexedFiles.getOrDefault(compare, null);
  }

  private static List<InputFile> getInputFiles(ModuleFileSystem moduleFileSystem) {
    List<InputFile> files = new ArrayList<>();
    moduleFileSystem.files(Python.KEY, InputFile.Type.MAIN).forEach(files::add);
    return Collections.unmodifiableList(files);
  }

  @Override
  void addFile(InputFile inputFile) throws IOException {
    super.addFile(inputFile);
    indexedFiles.put(inputFile.absolutePath(), inputFile);
  }

  @Override
  void removeFile(InputFile inputFile) {
    super.removeFile(inputFile);
    indexedFiles.remove(inputFile.absolutePath());
  }

  @Override
  public void process(ModuleFileEvent moduleFileEvent) {
    InputFile target = moduleFileEvent.getTarget();
    String language = target.language();
    if (language == null || !language.equals(Python.KEY)) {
      LOG.debug("Module file event for {} has been ignored because it's not a Python file.", target);
      return;
    }
    ModuleFileEvent.Type type = moduleFileEvent.getType();
    if (type.equals(ModuleFileEvent.Type.DELETED) || type.equals(ModuleFileEvent.Type.MODIFIED)) {
      removeFile(target);
    }
    if (type.equals(ModuleFileEvent.Type.CREATED) || type.equals(ModuleFileEvent.Type.MODIFIED)) {
      try {
        addFile(target);
      } catch (IOException e) {
        LOG.debug("Failed to load file \"{}\" ({}) to the project symbol table", target.filename(), type);
      }
    }
  }
}
