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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.caching.Caching;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.DependencyGraph;
import org.sonar.python.semantic.SymbolUtils;
import org.sonarsource.performance.measure.PerformanceMeasure;

public class SonarQubePythonIndexer extends PythonIndexer {

  private final List<InputFile> files;
  private final Set<InputFile> skippableFiles;
  private final CacheContext cacheContext;
  private static final Logger LOG = Loggers.get(SonarQubePythonIndexer.class);
  /**
   * Describes if an optimized analysis of unchanged by skipping some rules is enabled.
   * By default, the property is not set (null), leaving SQ/SC to decide whether to enable this behavior.
   * Setting it to true or false, forces the behavior from the analyzer independently of the server.
   */
  public static final String SONAR_CAN_SKIP_UNCHANGED_FILES_KEY = "sonar.python.skipUnchanged";
  public static final String IMPORT_MAP_CACHE_KEY_PREFIX = "python_import_map:";
  public static final String PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX = "python_project_symbol_table:";

  public SonarQubePythonIndexer(List<InputFile> files) {
    this.files = files;
    this.skippableFiles = new HashSet<>();
    this.cacheContext = new CacheContextImpl(); // received from constructor
  }

  @Override
  public void buildOnce(SensorContext context) {
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    LOG.debug("Input files for indexing: " + files);
    // information about caching / pr analysis can be retrieved from context
    boolean shouldUseCache = context.config().getBoolean(SONAR_CAN_SKIP_UNCHANGED_FILES_KEY).orElse(false) && cacheContext.isCacheEnabled();
    if (!shouldUseCache) {
      PerformanceMeasure.Duration duration = PerformanceMeasure.start("ProjectLevelSymbolTable");
      computeGlobalSymbols(files, context);
      duration.stop();
      return;
    }
    computeProjectLevelSymbolTableUsingCache(context);
  }

  private void computeProjectLevelSymbolTableUsingCache(SensorContext context) {
    Caching caching = new Caching(cacheContext);
    // retrieve list of imports graph from cache
    Map<InputFile, String> inputFileToFQN = files.stream().collect(Collectors.toMap(f -> f, f -> SymbolUtils.fullyQualifiedModuleName(packageName(f), f.filename())));
    // can we compute deleted files here through diff with what was saved? And include them to projectModuleFQNs / modifiedFiles
    Set<String> projectModulesFQNs = new HashSet<>(inputFileToFQN.values());
    DependencyGraph dependencyGraph = new DependencyGraph(Collections.emptyMap(), projectModulesFQNs); // use retrieved import map (if exists, else skip all this)
    // populate the skippableFiles map
    List<InputFile> modifiedFiles = new ArrayList<>();
    for (InputFile inputFile : files) {
      if (!inputFile.status().equals(InputFile.Status.SAME)) {
        modifiedFiles.add(inputFile);
      }
    }
    List<String> modulesFQN = modifiedFiles.stream()
      .map(f -> SymbolUtils.fullyQualifiedModuleName(packageName(f), f.filename()))
      .collect(Collectors.toList());
    Set<String> impactedModules = dependencyGraph.impactedModules(modulesFQN);
    for (InputFile inputFile : files) {
      if (!impactedModules.contains(inputFileToFQN.get(inputFile))) {
        skippableFiles.add(inputFile);
      }
    }
    // here we have a list of skippableFiles, need to retrieve their PST entry
    List<String> unchangedFilesThatFailedToRetrieve = new ArrayList<>();
    for (InputFile inputFile : skippableFiles) {
      String currFQN = inputFileToFQN.get(inputFile);
      Optional<Set<Descriptor>> descriptorsOptional = caching.readProjectLevelSymbolTableEntry(currFQN);
      if (descriptorsOptional.isPresent()) {
        Set<Descriptor> descriptors = descriptorsOptional.get();
        projectLevelSymbolTable().insertEntry(currFQN, descriptors);
        // retrieval of data went well: keep the entry in cache for next analysis
        cacheContext.getWriteCache().copyFromPrevious(IMPORT_MAP_CACHE_KEY_PREFIX + currFQN);
        cacheContext.getWriteCache().copyFromPrevious(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + currFQN);
      } else {
        unchangedFilesThatFailedToRetrieve.add(currFQN);
        modifiedFiles.add(inputFile);
      }
      Set<String> impactedModulesDueToFailedRetrieve = dependencyGraph.impactedModules(unchangedFilesThatFailedToRetrieve);
      skippableFiles.forEach(e -> {
        if (impactedModulesDueToFailedRetrieve.contains(inputFileToFQN.get(e))) {
          // these modules may be impacted as PST entries for their dependencies might have changed
          skippableFiles.remove(e);
        }
      });
    }
    // computes "globalSymbolsByModuleName"
    // even though we need to recompute IR and UCFG for all impacted files
    // we only need to recompute project symbol table entries for strictly modified files (no cross file dependency during computation of PST)
    computeGlobalSymbols(modifiedFiles, context);

    for (InputFile file : modifiedFiles) {
      // Update import map and PST entries for modified files
      String fileFQN = inputFileToFQN.get(file);
      caching.writeImportMapEntry(fileFQN, projectLevelSymbolTable().importsByModule().get(fileFQN));
      caching.writeProjectLevelSymbolTableEntry(fileFQN, projectLevelSymbolTable().descriptorsForModule(fileFQN));
    }
  }

  public void computeGlobalSymbols(List<InputFile> files, SensorContext context) {
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
  }

  @Override
  public boolean canBeScannedWithoutParsing(InputFile inputFile) {
    return skippableFiles.contains(inputFile);
  }
}
