/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.caching.Caching;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.DependencyGraph;
import org.sonar.python.semantic.SymbolUtils;
import org.sonarsource.performance.measure.PerformanceMeasure;

import static org.sonar.python.caching.Caching.IMPORTS_MAP_CACHE_KEY_PREFIX;
import static org.sonar.python.caching.Caching.PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX;

public class SonarQubePythonIndexer extends PythonIndexer {

  private final List<InputFile> mainFiles;
  private final List<InputFile> testFiles;
  private final Set<InputFile> skippableMainFiles;
  private final Set<InputFile> skippableTestFiles;
  private final CacheContext cacheContext;
  private static final Logger LOG = Loggers.get(SonarQubePythonIndexer.class);
  /**
   * Describes if an optimized analysis of unchanged by skipping some rules is enabled.
   * By default, the property is not set (null), leaving SQ/SC to decide whether to enable this behavior.
   * Setting it to true or false, forces the behavior from the analyzer independently of the server.
   */
  public static final String SONAR_CAN_SKIP_UNCHANGED_FILES_KEY = "sonar.python.skipUnchanged";

  public SonarQubePythonIndexer(List<InputFile> inputFiles, CacheContext cacheContext) {
    this.mainFiles = new ArrayList<>();
    this.testFiles = new ArrayList<>();
    inputFiles.forEach(f -> {
      if (f.type().equals(InputFile.Type.MAIN)) {
        mainFiles.add(f);
      } else {
        testFiles.add(f);
      }
    });
    this.skippableMainFiles = new HashSet<>();
    this.skippableTestFiles = new HashSet<>();
    this.cacheContext = cacheContext;
  }

  @Override
  public void buildOnce(SensorContext context) {
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    LOG.debug("Input files for indexing: " + mainFiles);
    boolean shouldUseCache = context.config().getBoolean(SONAR_CAN_SKIP_UNCHANGED_FILES_KEY).orElse(false) && cacheContext.isCacheEnabled();
    if (!shouldUseCache) {
      PerformanceMeasure.Duration duration = PerformanceMeasure.start("ProjectLevelSymbolTable");
      computeGlobalSymbols(mainFiles, context);
      duration.stop();
      return;
    }
    computeProjectLevelSymbolTableUsingCache(context);
  }

  private void computeProjectLevelSymbolTableUsingCache(SensorContext context) {
    LOG.info("Retrieving cached project level symbol table.");
    Caching caching = new Caching(cacheContext);
    Map<InputFile, String> inputFileToFQN = mainFiles.stream().collect(Collectors.toMap(f -> f, f -> SymbolUtils.fullyQualifiedModuleName(packageName(f), f.filename())));
    // Compute deleted files here through diff with what was saved and include them to projectModuleFQNs / modifiedFiles
    Set<String> projectModulesFQNs = new HashSet<>(inputFileToFQN.values());
    Map<String, Set<String>> importsByModule = new HashMap<>();
    List<InputFile> modifiedFiles = new ArrayList<>();
    List<String> impactfulFilesFQN = new ArrayList<>();
    for (InputFile inputFile : mainFiles) {
      String currFQN = inputFileToFQN.get(inputFile);
      if (!inputFile.status().equals(InputFile.Status.SAME)) {
        impactfulFilesFQN.add(currFQN);
        modifiedFiles.add(inputFile);
        continue;
      }
      Set<String> imports = caching.readImportMapEntry(currFQN);
      if (imports != null) {
        importsByModule.put(currFQN, imports);
      }
      Set<Descriptor> descriptors = caching.readProjectLevelSymbolTableEntry(currFQN);
      if (descriptors != null && imports != null) {
        saveRetrievedDescriptors(currFQN, descriptors);
      } else {
        // failed to retrieve the data: consider the file as impactful
        impactfulFilesFQN.add(currFQN);
        modifiedFiles.add(inputFile);
      }
    }
    DependencyGraph dependencyGraph = DependencyGraph.from(importsByModule, projectModulesFQNs);
    Set<String> impactedModulesFQN = dependencyGraph.impactedModules(impactfulFilesFQN);
    for (InputFile inputFile : mainFiles) {
      if (!impactedModulesFQN.contains(inputFileToFQN.get(inputFile))) {
        skippableMainFiles.add(inputFile);
      }
    }
    for (InputFile inputFile : testFiles) {
      if (inputFile.status().equals(InputFile.Status.SAME)) {
        skippableTestFiles.add(inputFile);
      }
    }
    LOG.info("Project level symbol table information needs to be computed for {} out of {} files.", impactfulFilesFQN.size(), mainFiles.size());
    LOG.info("Regular analysis will be performed on {} out of {} main files.", mainFiles.size()  - skippableMainFiles.size(), mainFiles.size());
    // Although we need to analyze all impacted files, we only need to recompute PST entries for modified files (no cross-file dependencies in the project symbol table)
    computeGlobalSymbols(modifiedFiles, context);
    saveGlobalSymbolsInCache(caching, inputFileToFQN, modifiedFiles);
  }

  private void saveGlobalSymbolsInCache(Caching caching, Map<InputFile, String> inputFileToFQN, List<InputFile> modifiedFiles) {
    for (InputFile file : modifiedFiles) {
      String fileFQN = inputFileToFQN.get(file);
      caching.writeImportsMapEntry(fileFQN, projectLevelSymbolTable().importsByModule().get(fileFQN));
      caching.writeProjectLevelSymbolTableEntry(fileFQN, projectLevelSymbolTable().descriptorsForModule(fileFQN));
    }
  }

  private void saveRetrievedDescriptors(String currFQN, Set<Descriptor> descriptors) {
    projectLevelSymbolTable().insertEntry(currFQN, descriptors);
    // Save the entry in cache for next analysis
    cacheContext.getWriteCache().copyFromPrevious(IMPORTS_MAP_CACHE_KEY_PREFIX + currFQN);
    cacheContext.getWriteCache().copyFromPrevious(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + currFQN);
  }

  public void computeGlobalSymbols(List<InputFile> files, SensorContext context) {
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
  }

  @Override
  public boolean canBeScannedWithoutParsing(InputFile inputFile) {
    return skippableMainFiles.contains(inputFile) || skippableTestFiles.contains(inputFile);
  }
}
