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
package org.sonar.plugins.python.indexer;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.caching.Caching;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.DependencyGraph;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.semantic.v2.typeshed.TypeShedDescriptorsProvider;
import org.sonarsource.performance.measure.PerformanceMeasure;

import static org.sonar.plugins.python.api.PythonVersionUtils.PYTHON_VERSION_KEY;

public class SonarQubePythonIndexer extends PythonIndexer {

  /**
   * Describes if an optimized analysis of unchanged by skipping some rules is enabled.
   * By default, the property is not set (null), leaving SQ/SC to decide whether to enable this behavior.
   * Setting it to true or false, forces the behavior from the analyzer independently of the server.
   */
  public static final String SONAR_CAN_SKIP_UNCHANGED_FILES_KEY = "sonar.python.skipUnchanged";
  private static final Logger LOG = LoggerFactory.getLogger(SonarQubePythonIndexer.class);

  private final Caching caching;
  private final Set<PythonInputFile> fullySkippableFiles = new HashSet<>();
  private final Set<PythonInputFile> partiallySkippableFiles = new HashSet<>();
  private final List<PythonInputFile> inputFiles = new ArrayList<>();
  private final Map<PythonInputFile, String> inputFileToFQN = new HashMap<>();

  public SonarQubePythonIndexer(List<PythonInputFile> inputFiles, CacheContext cacheContext, SensorContext context) {
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    this.caching = new Caching(cacheContext, getCacheVersion(context));
    inputFiles.forEach(f -> {
      this.inputFiles.add(f);
      inputFileToFQN.put(f, SymbolUtils.fullyQualifiedModuleName(packageName(f), f.wrappedFile().filename()));
    });
  }

  @Override
  public void buildOnce(SensorContext context) {
    LOG.debug("Input files for indexing: {}", inputFiles);
    collectPackageNames(inputFiles);
    if (shouldOptimizeAnalysis(context)) {
      computeGlobalSymbolsUsingCache(context);
      return;
    }
    PerformanceMeasure.Duration duration = PerformanceMeasure.start("ProjectLevelSymbolTable");
    computeGlobalSymbols(inputFiles, context);
    duration.stop();
  }

  @Override
  public void postAnalysis(SensorContext context) {
    if (caching.isCacheEnabled()) {
      Set<String> stubModules = projectLevelSymbolTable().typeShedDescriptorsProvider().stubModules();
      if (!stubModules.isEmpty()) {
        caching.writeTypeshedModules(stubModules);
      }
    }
  }

  private boolean shouldOptimizeAnalysis(SensorContext context) {
    return caching.isCacheEnabled()
      && (context.canSkipUnchangedFiles() || context.config().getBoolean(SONAR_CAN_SKIP_UNCHANGED_FILES_KEY).orElse(false))
      && caching.isCacheVersionUpToDate();
  }

  private void computeGlobalSymbolsUsingCache(SensorContext context) {
    loadTypeshedSymbols();
    projectLevelSymbolTable().typeShedDescriptorsProvider();
    LOG.info("Using cached data to retrieve global symbols.");
    Set<String> currentProjectModulesFQNs = new HashSet<>(inputFileToFQN.values());
    Set<String> deletedModulesFQNs = deletedModulesFQNs(currentProjectModulesFQNs);
    Set<String> allProjectFilesFQNs = Stream.concat(currentProjectModulesFQNs.stream(), deletedModulesFQNs.stream())
      .collect(Collectors.toSet());
    Map<String, Set<String>> importsByModule = new HashMap<>();
    // Deleted files are considered impactful to their dependents but will not be re-analyzed.
    List<PythonInputFile> impactfulFiles = new ArrayList<>();
    List<String> impactfulModulesFQNs = new ArrayList<>(deletedModulesFQNs);
    for (PythonInputFile inputFile : inputFiles) {
      String currFQN = inputFileToFQN.get(inputFile);
      boolean isUnimpacted = tryToUseCache(importsByModule, inputFile, currFQN);
      if (!isUnimpacted) {
        // Failed to retrieve some data: consider the file as impactful.
        impactfulFiles.add(inputFile);
        impactfulModulesFQNs.add(currFQN);
      } else {
        partiallySkippableFiles.add(inputFile);
      }
    }
    // Impacted modules are computed from both modified files and deleted ones.
    Set<String> impactedModulesFQN = DependencyGraph.from(importsByModule, allProjectFilesFQNs).impactedModules(impactfulModulesFQNs);
    inputFiles.stream().filter(f -> !impactedModulesFQN.contains(inputFileToFQN.get(f))).forEach(fullySkippableFiles::add);
    LOG.info(
      "Cached information of global symbols will be used for {} out of {} main files. Global symbols will be recomputed for the remaining files.",
      inputFiles.size() - impactfulFiles.size(),
      inputFiles.size());
    LOG.info("Fully optimized analysis can be performed for {} out of {} files.", fullySkippableFiles.size(), inputFiles.size());
    LOG.info("Partially optimized analysis can be performed for {} out of {} files.", partiallySkippableFiles.size(), inputFiles.size());
    // Although we need to analyze all impacted files, we only need to recompute global symbols for modified files (no cross-file dependencies
    // in the project symbol table)
    computeGlobalSymbols(impactfulFiles, context);
  }

  /*
   * In a full analysis, Typeshed symbols are loaded lazily depending on which module is encountered during parsing.
   * SonarSecurity needs all Typeshed symbols used in the project to be properly loaded.
   * For that reason, we load all symbols that were used in the previous analysis upfront, even if the file using them will not be parsed.
   */
  private void loadTypeshedSymbols() {
    TypeShedDescriptorsProvider typeshedReader = projectLevelSymbolTable().typeShedDescriptorsProvider();
    Set<String> typeShedModules = caching.readTypeshedModules();
    typeShedModules.forEach(typeshedReader::descriptorsForModule);
  }

  private boolean tryToUseCache(Map<String, Set<String>> importsByModule, PythonInputFile inputFile, String currFQN) {
    if (!fileIsUnchanged(inputFile)) {
      return false;
    }

    Set<String> imports = caching.readImportMapEntry(inputFile.wrappedFile().key());
    if (imports != null) {
      importsByModule.put(currFQN, imports);
    }
    Set<Descriptor> descriptors = caching.readProjectLevelSymbolTableEntry(inputFile.wrappedFile().key());
    if (descriptors != null && imports != null) {
      saveRetrievedDescriptors(inputFile.wrappedFile().key(), descriptors, caching);
      return true;
    }

    return false;
  }

  private boolean fileIsUnchanged(PythonInputFile inputFile) {
    if (!inputFile.wrappedFile().status().equals(InputFile.Status.SAME)) {
      return false;
    }
    byte[] fileHash = caching.readFileContentHash(inputFile.wrappedFile().key());
    // InputFile.Status is not reliable in some cases
    // We use the hash of the file's content to double-check the content is the same.
    var fileInputHash = inputFile.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8);
    return MessageDigest.isEqual(fileHash, fileInputHash);
  }

  private void saveRetrievedDescriptors(String fileKey, Set<Descriptor> descriptors, Caching caching) {
    projectLevelSymbolTable().insertEntry(fileKey, descriptors);
    caching.copyFromPrevious(fileKey);
  }

  public void computeGlobalSymbols(List<PythonInputFile> files, SensorContext context) {
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
    if (caching.isCacheEnabled()) {
      saveGlobalSymbolsInCache(files);
      saveMainFilesListInCache(new HashSet<>(inputFileToFQN.values()));
      // Information on used Typeshed stubs needs to be done at the end of the analysis, as it is not computed during indexing anymore
      caching.writeCacheVersion();
    }
  }

  private void saveGlobalSymbolsInCache(List<PythonInputFile> files) {
    for (PythonInputFile inputFile : files) {
      String moduleFQN = inputFileToFQN.get(inputFile);
      Set<Descriptor> descriptors = projectLevelSymbolTable().descriptorsForModule(moduleFQN);
      Set<String> imports = projectLevelSymbolTable().importsByModule().get(moduleFQN);
      if (descriptors != null && imports != null) {
        // Descriptors/imports map may be null if the file failed to parse.
        // We don't try to save information in the cache in that case.
        writeContentHashToCache(inputFile);

        caching.writeProjectLevelSymbolTableEntry(inputFile.wrappedFile().key(), descriptors);
        caching.writeImportsMapEntry(inputFile.wrappedFile().key(), imports);
      }
    }
  }

  private void writeContentHashToCache(PythonInputFile inputFile) {
    var contentHash = inputFile.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8);
    caching.writeFileContentHash(inputFile.wrappedFile().key(), contentHash);
  }

  private Set<String> deletedModulesFQNs(Set<String> projectModulesFQNs) {
    Set<String> previousAnalysisModulesFQNs = caching.readFilesList();
    previousAnalysisModulesFQNs.removeAll(projectModulesFQNs);
    return previousAnalysisModulesFQNs;
  }

  private void saveMainFilesListInCache(Set<String> modulesFQN) {
    caching.writeFilesList(new ArrayList<>(modulesFQN));
  }

  @Override
  public boolean canBePartiallyScannedWithoutParsing(PythonInputFile inputFile) {
    return partiallySkippableFiles.contains(inputFile) || fullySkippableFiles.contains(inputFile);
  }

  @Override
  public boolean canBeFullyScannedWithoutParsing(PythonInputFile inputFile) {
    return fullySkippableFiles.contains(inputFile);
  }

  @Override
  public CacheContext cacheContext() {
    return caching.cacheContext();
  }

  private static String getCacheVersion(SensorContext context) {
    String implementationVersion = getImplementationVersion(SonarQubePythonIndexer.class);
    var pythonVersions = context.config().getStringArray(PYTHON_VERSION_KEY);
    if (pythonVersions.length == 0) {
      return implementationVersion;
    }
    return implementationVersion + ";" + String.join(",", pythonVersions);
  }

  private static String getImplementationVersion(Class<?> cls) {
    String implementationVersion = cls.getPackage().getImplementationVersion();
    if (implementationVersion == null) {
      LOG.warn("Implementation version of the Python plugin not found. Cached data may not be invalidated properly, which may lead to inaccurate analysis results.");
      return "unknownPluginVersion";
    }
    return implementationVersion;
  }
}
