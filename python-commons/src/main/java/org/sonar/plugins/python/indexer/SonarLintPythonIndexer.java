/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.PythonInputFileImpl;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.caching.PythonReadCacheImpl;
import org.sonar.python.caching.PythonWriteCacheImpl;
import org.sonarsource.api.sonarlint.SonarLintSide;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileEvent;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileListener;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileSystem;

import static org.sonar.python.semantic.SymbolUtils.resolvedPath;

@SonarLintSide(lifespan = "MODULE")
public class SonarLintPythonIndexer extends PythonIndexer implements ModuleFileListener {

  private final ModuleFileSystem moduleFileSystem;

  private CacheContext cacheContext;
  private @Nullable FileSystem fileSystem;
  private @Nullable Configuration configuration;
  private final PackageRootState packageRootState = new PackageRootState();
  private final Map<String, InputFile> knownMainFiles = new ConcurrentHashMap<>();
  private final Map<String, InputFile> indexedFiles = new ConcurrentHashMap<>();
  private static final Logger LOG = LoggerFactory.getLogger(SonarLintPythonIndexer.class);
  private boolean shouldBuildProjectSymbolTable = true;
  private boolean indexingEnabled = true;
  private static final long DEFAULT_MAX_LINES_FOR_INDEXING = 300_000;
  private static final String MAX_LINES_PROPERTY = "sonar.python.sonarlint.indexing.maxlines";

  public SonarLintPythonIndexer(ModuleFileSystem moduleFileSystem, ProjectConfigurationBuilder projectConfigurationBuilder) {
    super(projectConfigurationBuilder);
    this.moduleFileSystem = moduleFileSystem;
  }

  @Override
  public void buildOnce(SensorContext context) {
    if (!shouldBuildProjectSymbolTable) {
      return;
    }
    this.fileSystem = context.fileSystem();
    this.configuration = context.config();
    List<InputFile> rootContributorFiles = getPackageRootContributorFiles(moduleFileSystem);
    setPackageResolutionResult(PackageRootResolver.resolve(fileSystem, configuration, rootContributorFiles));
    shouldBuildProjectSymbolTable = false;
    List<PythonInputFile> files = getInputFiles(moduleFileSystem);
    collectPackageNames(files);
    long nLines = files.stream().map(PythonInputFile::wrappedFile).map(InputFile::lines).mapToLong(Integer::longValue).sum();
    long maxLinesForIndexing = context.config().getLong(MAX_LINES_PROPERTY).orElse(DEFAULT_MAX_LINES_FOR_INDEXING);
    if (nLines > maxLinesForIndexing) {
      // Avoid performance issues for large projects
      LOG.debug("Project symbol table deactivated due to project size (total number of lines is {}, maximum for indexing is {})", nLines, maxLinesForIndexing);
      LOG.debug("Update \"sonar.python.sonarlint.indexing.maxlines\" to set a different limit.");
      indexingEnabled = false;
      return;
    }
    files.stream()
      .map(PythonInputFile::wrappedFile)
      .forEach(file -> knownMainFiles.put(file.absolutePath(), file));
    packageRootState.initialize(rootContributorFiles);
    LOG.debug("Input files for indexing: {}", files);
    // computes "globalSymbolsByModuleName"
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
  }

  @Override
  public void postAnalysis(SensorContext context) {
    // no op
  }

  // SonarLintCache has to be set lazily because SonarLintPythonIndex is injected in the PythonSensor
  @Override
  public void setSonarLintCache(@Nullable SonarLintCache sonarLintCache) {
    if (sonarLintCache != null) {
      // ^This null check is defensive.
      // In practice, a SonarLintCache instance should always be available when a SonarLintPythonIndexer is available.
      // See also PythonPlugin::SonarLintPluginAPIManager::addSonarlintPythonIndexer.
      this.cacheContext = new CacheContextImpl(true, new PythonWriteCacheImpl(sonarLintCache), new PythonReadCacheImpl(sonarLintCache));
    }
  }

  @Override
  public InputFile getFileWithId(String fileId) {
    String compare = fileId.replace("\\", "/");
    return indexedFiles.getOrDefault(compare, null);
  }

  @Override
  public CacheContext cacheContext() {
    return cacheContext != null ? cacheContext : CacheContextImpl.dummyCache();
  }

  private static List<PythonInputFile> getInputFiles(ModuleFileSystem moduleFileSystem) {
    List<PythonInputFile> files = new ArrayList<>();
    moduleFileSystem.files(Python.KEY, InputFile.Type.MAIN).map(PythonInputFileImpl::new).forEach(files::add);
    return Collections.unmodifiableList(files);
  }

  private static List<InputFile> getPackageRootContributorFiles(ModuleFileSystem moduleFileSystem) {
    return moduleFileSystem.files()
      .filter(file -> Python.KEY.equals(file.language()))
      .toList();
  }

  @VisibleForTesting
  boolean refreshPackageRoots() {
    if (fileSystem == null || configuration == null) {
      return false;
    }
    try {
      PackageResolutionResult result = PackageRootResolver.resolve(fileSystem, configuration, packageRootState.contributorFiles());
      boolean rootsChanged = !packageRoots.equals(result.roots());
      setPackageResolutionResult(result);
      packageRootState.onResolutionSucceeded(this::packageRootCandidate);
      return rootsChanged;
    } catch (RuntimeException e) {
      // This module-scoped indexer survives failed events. Contributor state may already reflect the event while
      // resolved roots or candidates do not, so require full resolution before handling another structural event.
      packageRootState.onResolutionFailed();
      throw e;
    }
  }

  private @Nullable Path packageRootCandidate(InputFile file) {
    if (fileSystem == null || file.file().getParentFile() == null) {
      return null;
    }
    String candidateRoot = PackageRootResolver.adjustPackageRoot(file.file().getParentFile(), fileSystem.baseDir());
    return resolvedPath(new File(candidateRoot));
  }

  private void rebuildProjectIndex() {
    clearProjectIndex();
    List<PythonInputFile> filesToRebuild = knownMainFiles.values().stream()
      .<PythonInputFile>map(PythonInputFileImpl::new)
      .toList();
    collectPackageNames(filesToRebuild);
    for (PythonInputFile inputFile : filesToRebuild) {
      try {
        super.addFile(inputFile);
        indexedFiles.put(inputFile.wrappedFile().absolutePath(), inputFile.wrappedFile());
      } catch (Exception e) {
        LOG.debug("Failed to re-index file \"{}\" during project index rebuild", inputFile.wrappedFile().filename(), e);
      }
    }
    recreateProjectLevelTypeTable();
  }

  private PackageRootImpact packageRootImpact(PythonInputFile target, ModuleFileEvent.Type type) {
    Path candidateRoot = type == ModuleFileEvent.Type.CREATED ? packageRootCandidate(target.wrappedFile()) : null;
    if (isPackageConfigurationFile(target.wrappedFile().filename())) {
      return new PackageRootImpact(true, candidateRoot);
    }
    if (type == ModuleFileEvent.Type.MODIFIED) {
      return new PackageRootImpact(false, null);
    }
    if (packageRootState.resolutionRetryRequired() || isPackageInitFile(target.wrappedFile().filename())) {
      return new PackageRootImpact(true, candidateRoot);
    }
    if (type == ModuleFileEvent.Type.CREATED) {
      return createdFilePackageRootImpact(target.wrappedFile(), candidateRoot);
    }
    if (type == ModuleFileEvent.Type.DELETED) {
      return deletedFilePackageRootImpact(target.wrappedFile());
    }
    return new PackageRootImpact(true, candidateRoot);
  }

  private PackageRootImpact createdFilePackageRootImpact(InputFile target, @Nullable Path candidateRoot) {
    boolean canImpactRoots = candidateRoot == null
      || canIntroduceConventionalFolder(target)
      || (!isCoveredByBaseDirFallback(target) && !isResolvedPackageRoot(candidateRoot));
    return new PackageRootImpact(canImpactRoots, candidateRoot);
  }

  private PackageRootImpact deletedFilePackageRootImpact(InputFile target) {
    File parentDirectory = target.file().getParentFile();
    if (parentDirectory == null || !parentDirectory.isDirectory()) {
      return new PackageRootImpact(true, null);
    }
    if (isCoveredByBaseDirFallback(target)) {
      return new PackageRootImpact(false, null);
    }
    return new PackageRootImpact(packageRootState.deletionCanImpactRoots(target.absolutePath(), this::isResolvedPackageRoot), null);
  }

  private static boolean isPackageConfigurationFile(String filename) {
    return "setup.py".equals(filename) || "pyproject.toml".equals(filename);
  }

  private static boolean isPackageInitFile(String filename) {
    return "__init__.py".equals(filename);
  }

  private boolean canIntroduceConventionalFolder(InputFile target) {
    if (fileSystem == null || packageResolutionResult == null) {
      return true;
    }
    PackageResolutionResult.PrimaryResolutionMethod method = packageResolutionResult.method();
    if (method != PackageResolutionResult.PrimaryResolutionMethod.BASE_DIR
      && method != PackageResolutionResult.PrimaryResolutionMethod.SONAR_SOURCES) {
      return false;
    }
    Path targetPath = resolvedPath(target.file());
    Path baseDirPath = resolvedPath(fileSystem.baseDir());
    return PackageRootResolver.CONVENTIONAL_FOLDERS.stream()
      .map(baseDirPath::resolve)
      .anyMatch(targetPath::startsWith);
  }

  private boolean isCoveredByBaseDirFallback(InputFile target) {
    return fileSystem != null
      && packageResolutionResult != null
      && packageResolutionResult.method() == PackageResolutionResult.PrimaryResolutionMethod.BASE_DIR
      && isUnderResolvedPackageRoot(resolvedPath(target.file()));
  }

  /**
   * Tracks the Python files that contribute to package-root resolution and caches each file's candidate legacy root
   * (the root obtained by walking up its parent directories while they contain an {@code __init__.py} file).
   *
   * <p>The candidate cache allows ordinary file events to skip full package-root resolution when the event cannot add
   * or remove a resolved root. Structural changes such as creating or deleting an {@code __init__.py} invalidate the
   * cache so that subsequent events remain conservative.
   */
  private static final class PackageRootState {
    private final Map<String, InputFile> contributorFiles = new ConcurrentHashMap<>();
    private final Map<String, Path> candidates = new ConcurrentHashMap<>();
    private boolean candidatesValid;
    private boolean resolutionRetryRequired;

    void initialize(List<InputFile> files) {
      contributorFiles.clear();
      files.forEach(file -> contributorFiles.put(file.absolutePath(), file));
      candidates.clear();
      candidatesValid = false;
      resolutionRetryRequired = false;
    }

    List<InputFile> contributorFiles() {
      return List.copyOf(contributorFiles.values());
    }

    /** Marks that root resolution failed so subsequent structural file events retry it instead of using cached state. */
    void onResolutionFailed() {
      resolutionRetryRequired = true;
    }

    /**
     * Synchronizes the candidate cache after successful root resolution and allows structural file events to use it
     * again. This must only be called after the new resolution result has been installed.
     */
    void onResolutionSucceeded(Function<InputFile, Path> candidateResolver) {
      if (candidatesValid) {
        resolutionRetryRequired = false;
        return;
      }
      candidates.clear();
      for (Map.Entry<String, InputFile> entry : contributorFiles.entrySet()) {
        Path candidate = candidateResolver.apply(entry.getValue());
        if (candidate == null) {
          resolutionRetryRequired = false;
          return;
        }
        candidates.put(entry.getKey(), candidate);
      }
      candidatesValid = true;
      resolutionRetryRequired = false;
    }

    boolean resolutionRetryRequired() {
      return resolutionRetryRequired;
    }

    boolean deletionCanImpactRoots(String absolutePath, Predicate<Path> isResolvedPackageRoot) {
      if (!candidatesValid) {
        return true;
      }
      Path previousCandidate = candidates.get(absolutePath);
      if (previousCandidate == null || !isResolvedPackageRoot.test(previousCandidate)) {
        return previousCandidate == null;
      }
      return candidates.entrySet().stream()
        .noneMatch(entry -> !entry.getKey().equals(absolutePath) && entry.getValue().equals(previousCandidate));
    }

    void update(PythonInputFile target, ModuleFileEvent.Type type, @Nullable Path candidate) {
      InputFile file = target.wrappedFile();
      String absolutePath = file.absolutePath();
      if (isPackageInitFile(file.filename())
        && (type == ModuleFileEvent.Type.CREATED || type == ModuleFileEvent.Type.DELETED)) {
        candidatesValid = false;
      }
      if (isCreatedOrModified(type)) {
        contributorFiles.put(absolutePath, file);
        if (type == ModuleFileEvent.Type.CREATED) {
          if (candidate != null) {
            candidates.put(absolutePath, candidate);
          } else {
            candidatesValid = false;
          }
        }
      } else if (type == ModuleFileEvent.Type.DELETED) {
        contributorFiles.remove(absolutePath);
        candidates.remove(absolutePath);
      }
    }
  }

  private record PackageRootImpact(boolean canImpactRoots, @Nullable Path candidateRoot) {
  }

  private void updateKnownMainFile(PythonInputFile target, ModuleFileEvent.Type type, boolean mainPythonFile) {
    String absolutePath = target.wrappedFile().absolutePath();
    boolean wasKnownMainFile = knownMainFiles.containsKey(absolutePath);
    boolean shouldRemoveFromIndex = type.equals(ModuleFileEvent.Type.DELETED)
      || type.equals(ModuleFileEvent.Type.MODIFIED)
      || !mainPythonFile;
    if (wasKnownMainFile && shouldRemoveFromIndex) {
      removeFile(target);
    }
    if (mainPythonFile && isCreatedOrModified(type)) {
      knownMainFiles.put(absolutePath, target.wrappedFile());
    } else if (type.equals(ModuleFileEvent.Type.DELETED) || !mainPythonFile) {
      knownMainFiles.remove(absolutePath);
    }
  }

  private void addFileIfCreatedOrModified(PythonInputFile target, ModuleFileEvent.Type type) throws IOException {
    if (isCreatedOrModified(type)) {
      addFile(target);
    }
  }

  private static boolean isCreatedOrModified(ModuleFileEvent.Type type) {
    return type.equals(ModuleFileEvent.Type.CREATED) || type.equals(ModuleFileEvent.Type.MODIFIED);
  }

  @Override
  protected void addFile(PythonInputFile inputFile) throws IOException {
    super.addFile(inputFile);
    indexedFiles.put(inputFile.wrappedFile().absolutePath(), inputFile.wrappedFile());
    recreateProjectLevelTypeTable();
  }

  @Override
  void removeFile(PythonInputFile inputFile) {
    super.removeFile(inputFile);
    indexedFiles.remove(inputFile.wrappedFile().absolutePath());
    recreateProjectLevelTypeTable();
  }

  @Override
  protected void clearProjectIndex() {
    super.clearProjectIndex();
    indexedFiles.clear();
  }

  @Override
  public synchronized void process(ModuleFileEvent moduleFileEvent) {
    PythonInputFile target = new PythonInputFileImpl(moduleFileEvent.getTarget());
    String language = target.wrappedFile().language();
    boolean packageConfigurationFile = isPackageConfigurationFile(target.wrappedFile().filename());
    boolean pythonFile = Python.KEY.equals(language);
    boolean mainPythonFile = pythonFile && target.wrappedFile().type() == InputFile.Type.MAIN;
    if (!pythonFile && !packageConfigurationFile) {
      LOG.debug("Module file event for {} has been ignored because it's not a Python file.", target);
      return;
    }
    if (!indexingEnabled) {
      return;
    }
    ModuleFileEvent.Type type = moduleFileEvent.getType();
    try {
      PackageRootImpact packageRootImpact = packageRootImpact(target, type);
      if (pythonFile) {
        packageRootState.update(target, type, packageRootImpact.candidateRoot());
        updateKnownMainFile(target, type, mainPythonFile);
      }

      if (packageRootImpact.canImpactRoots() && refreshPackageRoots()) {
        rebuildProjectIndex();
      } else if (mainPythonFile) {
        addFileIfCreatedOrModified(target, type);
      }
    } catch (Exception e) {
      indexedFiles.remove(target.wrappedFile().absolutePath());
      LOG.debug("Failed to load file \"{}\" ({}) to the project symbol table", target.wrappedFile().filename(), type, e);
    }
  }
}
