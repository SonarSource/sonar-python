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
package org.sonar.plugins.python.indexer;

import com.sonar.sslr.api.AstNode;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.Scanner;
import org.sonar.plugins.python.SonarQubePythonFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.PythonTreeMaker;

import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public abstract class PythonIndexer {

  private static final Logger LOG = LoggerFactory.getLogger(PythonIndexer.class);

  protected String projectBaseDirAbsolutePath;

  private final Map<URI, String> packageNames = new HashMap<>();
  private final PythonParser parser = PythonParser.create();
  private final ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

  public ProjectLevelSymbolTable projectLevelSymbolTable() {
    return projectLevelSymbolTable;
  }


  public String packageName(PythonInputFile inputFile) {
    if (!packageNames.containsKey(inputFile.wrappedFile().uri())) {
      String name = pythonPackageName(inputFile.wrappedFile().file(), projectBaseDirAbsolutePath);
      packageNames.put(inputFile.wrappedFile().uri(), name);
      projectLevelSymbolTable.addProjectPackage(name);
    }
    return packageNames.get(inputFile.wrappedFile().uri());
  }

  public void collectPackageNames(List<PythonInputFile> inputFiles) {
    for (PythonInputFile inputFile : inputFiles) {
      String packageName = pythonPackageName(inputFile.wrappedFile().file(), projectBaseDirAbsolutePath);
      projectLevelSymbolTable.addProjectPackage(packageName);
    }
  }

  void removeFile(PythonInputFile inputFile) {
    String packageName = packageNames.get(inputFile.wrappedFile().uri());
    String filename = inputFile.wrappedFile().filename();
    if (packageName == null) {
      LOG.debug("Failed to remove file \"{}\" from project-level symbol table (file not indexed)", filename);
      return;
    }
    packageNames.remove(inputFile.wrappedFile().uri());
    projectLevelSymbolTable.removeModule(packageName, filename);
  }

  void addFile(PythonInputFile inputFile) throws IOException {
    AstNode astNode = parser.parse(inputFile.wrappedFile().contents());
    FileInput astRoot = new PythonTreeMaker().fileInput(astNode);
    String packageName = pythonPackageName(inputFile.wrappedFile().file(), projectBaseDirAbsolutePath);
    packageNames.put(inputFile.wrappedFile().uri(), packageName);
    projectLevelSymbolTable.addProjectPackage(packageName);
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile.wrappedFile());
    projectLevelSymbolTable.addModule(astRoot, packageName, pythonFile);
  }

  public abstract void buildOnce(SensorContext context);

  public abstract void postAnalysis(SensorContext context);

  public void setSonarLintCache(@Nullable SonarLintCache sonarLintCache) {
    // no op by default
  }

  @CheckForNull
  public InputFile getFileWithId(String fileId) {
    // no op by default
    return null;
  }

  /**
   * @param inputFile
   * @return true if a file is partially skippable, false otherwise
   * We consider a file to be partially skippable if it is unchanged, but may depend on impacted files.
   * Regular Python rules will not run on such files.
   * Security UCFGs and DBD IRs will be regenerated for them if they do depend on impacted files.
   * In such case, these files will still need to be parsed when Security or DBD rules are enabled.
   */
  public boolean canBePartiallyScannedWithoutParsing(PythonInputFile inputFile) {
    return false;
  }

  /**
   * @param inputFile
   * @return true if a file is fully skippable, false otherwise
   * We consider a file to be fully skippable if it is unchanged and does NOT depend on any impacted file.
   * Regular Python rules will not run on these files. Security UCFGs and DBD IRs will be retrieved from the cache.
   * These files will not be parsed.
   */
  public boolean canBeFullyScannedWithoutParsing(PythonInputFile inputFile) {
    return false;
  }

  public abstract CacheContext cacheContext();

  class GlobalSymbolsScanner extends Scanner {

    protected GlobalSymbolsScanner(SensorContext context) {
      super(context);
    }

    @Override
    protected String name() {
      return "global symbols computation";
    }

    @Override
    protected void scanFile(PythonInputFile inputFile) throws IOException {
      // Global Symbol Table is deactivated for Notebooks see: SONARPY-2021
      if (inputFile.kind() == PythonInputFile.Kind.PYTHON) {
        addFile(inputFile);
      }
    }

    @Override
    protected void processException(Exception e, PythonInputFile file) {
      LOG.debug("Unable to construct project-level symbol table for file: {}", file);
      LOG.debug(e.getMessage());
    }
  }
}
