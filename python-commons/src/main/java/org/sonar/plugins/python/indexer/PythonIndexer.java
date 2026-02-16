/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import com.google.common.annotations.VisibleForTesting;
import com.sonar.sslr.api.AstNode;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.Scanner;
import org.sonar.plugins.python.SonarQubePythonFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.project.config.SignatureBasedAwsLambdaHandlersCollector;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typetable.CachedTypeTable;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.typetable.TypeTable;
import org.sonar.python.tree.PythonTreeMaker;

import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public abstract class PythonIndexer {

  private static final Logger LOG = LoggerFactory.getLogger(PythonIndexer.class);

  protected String projectBaseDirAbsolutePath;
  protected List<String> packageRoots = List.of();

  private final Map<URI, String> packageNames = new ConcurrentHashMap<>();
  private final Supplier<PythonParser> parserSupplier = PythonParser::create;

  private final ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
  private TypeTable projectLevelTypeTable;

  private final SignatureBasedAwsLambdaHandlersCollector signatureBasedAwsLambdaHandlersCollector = new SignatureBasedAwsLambdaHandlersCollector();
  private final ProjectConfigurationBuilder projectConfigurationBuilder;
  private @Nullable NamespacePackageTelemetry namespacePackageTelemetry;

  protected PythonIndexer(ProjectConfigurationBuilder projectConfigurationBuilder) {
    this.projectConfigurationBuilder = projectConfigurationBuilder;
    recreateProjectLevelTypeTable();
  }

  public ProjectLevelSymbolTable projectLevelSymbolTable() {
    return projectLevelSymbolTable;
  }

  public ProjectConfiguration projectConfig() {
    return projectConfigurationBuilder.build();
  }

  public TypeTable projectLevelTypeTable() {
    return projectLevelTypeTable;
  }

  protected void recreateProjectLevelTypeTable() {
    projectLevelTypeTable = new CachedTypeTable(new ProjectLevelTypeTable(projectLevelSymbolTable));
  }

  public String packageName(PythonInputFile inputFile) {
    if (!packageNames.containsKey(inputFile.wrappedFile().uri())) {
      String name = pythonPackageName(inputFile.wrappedFile().file(), packageRoots, projectBaseDirAbsolutePath);
      packageNames.put(inputFile.wrappedFile().uri(), name);
      projectLevelSymbolTable.addProjectPackage(name);
    }
    return packageNames.get(inputFile.wrappedFile().uri());
  }

  public List<String> packageRoots() {
    return packageRoots;
  }

  /**
   * Resolves package root directories for the project.
   *
   * <p>Attempts to extract source roots from pyproject.toml and setup.py build system configurations.
   * Falls back to sonar.sources property, conventional folders (src/, lib/), or the project base directory.
   *
   * @param context the sensor context containing filesystem and configuration
   * @return list of resolved package root absolute paths
   */
  protected List<String> resolvePackageRoots(SensorContext context) {
    FileSystem fileSystem = context.fileSystem();
    File baseDir = fileSystem.baseDir();
    List<String> extractedRoots = extractSourceRoots(fileSystem);
    List<String> resolvedRoots = PackageRootResolver.resolve(extractedRoots, context.config(), baseDir);
    List<String> adjustedRoots = resolvedRoots.stream()
      .map(root -> adjustPackageRoot(new File(root), baseDir))
      .distinct()
      .toList();
    if (!extractedRoots.isEmpty()) {
      LOG.debug("Resolved package roots from build configuration: {}", resolvedRoots);
    } else if (!resolvedRoots.isEmpty() && !resolvedRoots.get(0).equals(baseDir.getAbsolutePath())) {
      LOG.debug("Resolved package roots from fallback (sonar.sources or conventional folders): {}", resolvedRoots);
    }
    return adjustedRoots;
  }

  /**
   * Extracts source root directories from build configuration files (pyproject.toml and setup.py).
   *
   * @param fileSystem the file system to search for configuration files
   * @return list of extracted source root paths (relative), or empty list if none found
   */
  private static List<String> extractSourceRoots(FileSystem fileSystem) {
    List<String> pyprojectRoots = extractSourceRootsFromPyProjectToml(fileSystem);
    List<String> setupPyRoots = extractSourceRootsFromSetupPy(fileSystem);

    return Stream.concat(pyprojectRoots.stream(), setupPyRoots.stream())
      .distinct()
      .toList();
  }

  /**
   * Adjusts a package root by walking up the directory tree if it contains __init__.py.
   *
   * <p>If the root directory contains __init__.py, it's part of a package, not the package root.
   * We walk up to find the first parent directory without __init__.py.
   *
   * @param root the potential package root directory
   * @param baseDir the project base directory (we don't walk above this)
   * @return the adjusted package root absolute path
   */
  @VisibleForTesting
  static String adjustPackageRoot(File root, File baseDir) {
    File current = root;
    String baseDirPath = baseDir.getAbsolutePath();
    while (current != null && !current.getAbsolutePath().equals(baseDirPath)) {
      File initFile = new File(current, "__init__.py");
      if (!initFile.exists()) {
        break;
      }
      current = current.getParentFile();
    }
    if (current == null) {
      return baseDirPath;
    }
    return current.getAbsolutePath();
  }

  /**
   * Recursively finds files with the given filename in the project directory.
   *
   * @param fileSystem the file system to search in
   * @param filename the filename to search for (e.g., "pyproject.toml", "setup.py")
   * @return stream of matching files
   */
  private static Stream<File> findFilesRecursively(FileSystem fileSystem, String filename) {
    try {
      return Files.walk(fileSystem.baseDir().toPath())
        .filter(Files::isRegularFile)
        .filter(path -> filename.equals(path.getFileName().toString()))
        .map(Path::toFile);
    } catch (IOException e) {
      return Stream.empty();
    }
  }

  /**
   * Extracts source root directories from pyproject.toml files in the project.
   *
   * @param fileSystem the file system to search for pyproject.toml
   * @return list of extracted source root paths (relative), or empty list if none found
   */
  private static List<String> extractSourceRootsFromPyProjectToml(FileSystem fileSystem) {
    return findFilesRecursively(fileSystem, "pyproject.toml")
      .flatMap(file -> PyProjectTomlSourceRoots.extract(file).stream())
      .distinct()
      .toList();
  }

  /**
   * Extracts source root directories from setup.py files in the project.
   *
   * @param fileSystem the file system to search for setup.py
   * @return list of extracted source root paths (relative), or empty list if none found
   */
  private static List<String> extractSourceRootsFromSetupPy(FileSystem fileSystem) {
    return findFilesRecursively(fileSystem, "setup.py")
      .flatMap(file -> SetupPySourceRoots.extract(file).stream())
      .distinct()
      .toList();
  }

  public void collectPackageNames(List<PythonInputFile> inputFiles) {
    for (PythonInputFile inputFile : inputFiles) {
      String packageName = pythonPackageName(inputFile.wrappedFile().file(), packageRoots, projectBaseDirAbsolutePath);
      projectLevelSymbolTable.addProjectPackage(packageName);
    }
  }

  protected void analyzeNamespacePackages(ProjectTree projectTree) {
    try {
      this.namespacePackageTelemetry = new NamespacePackageAnalyzer().analyze(projectTree);
    } catch (Exception e) {
      // Ensures that telemetry cannot crash the analysis
      LOG.warn("Failed to analyze namespace packages", e);
    }
  }

  @CheckForNull
  public NamespacePackageTelemetry namespacePackageTelemetry() {
    return namespacePackageTelemetry;
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
    projectConfigurationBuilder.removePackageAwsLambdaHandlers(packageName);
  }

  void addFile(PythonInputFile inputFile) throws IOException {
    AstNode astNode = parserSupplier.get().parse(inputFile.wrappedFile().contents());
    FileInput astRoot = new PythonTreeMaker().fileInput(astNode);
    String packageName = pythonPackageName(inputFile.wrappedFile().file(), packageRoots, projectBaseDirAbsolutePath);
    packageNames.put(inputFile.wrappedFile().uri(), packageName);
    projectLevelSymbolTable.addProjectPackage(packageName);
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile.wrappedFile());
    projectLevelSymbolTable.addModule(astRoot, packageName, pythonFile);
    signatureBasedAwsLambdaHandlersCollector.collect(projectConfigurationBuilder, astRoot, packageName);
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
    protected void logStart(int numThreads) {
      if (numThreads != 1) {
        LOG.debug("Scanning global symbols in {} threads", numThreads);
      }
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
      LOG.debug("Unable to construct project-level symbol table for file: {}", file, e);
    }

  }
}
