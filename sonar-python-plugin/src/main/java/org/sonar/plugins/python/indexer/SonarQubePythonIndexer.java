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
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.caching.Caching;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.DependencyGraph;
import org.sonar.python.semantic.SymbolUtils;
import org.sonarsource.performance.measure.PerformanceMeasure;

public class SonarQubePythonIndexer extends PythonIndexer {

  /**
   * Describes if an optimized analysis of unchanged by skipping some rules is enabled.
   * By default, the property is not set (null), leaving SQ/SC to decide whether to enable this behavior.
   * Setting it to true or false, forces the behavior from the analyzer independently of the server.
   */
  public static final String SONAR_CAN_SKIP_UNCHANGED_FILES_KEY = "sonar.python.skipUnchanged";
  private static final Logger LOG = Loggers.get(SonarQubePythonIndexer.class);

  private final Caching caching;
  private final Set<InputFile> skippableFiles = new HashSet<>();
  private final List<InputFile> mainFiles = new ArrayList<>();
  private final List<InputFile> testFiles = new ArrayList<>();
  private final Map<InputFile, String> inputFileToFQN = new HashMap<>();

  public SonarQubePythonIndexer(List<InputFile> inputFiles, CacheContext cacheContext) {
    inputFiles.forEach(f -> {
      if (f.type().equals(InputFile.Type.MAIN)) {
        mainFiles.add(f);
      } else {
        testFiles.add(f);
      }
    });
    this.caching = new Caching(cacheContext);
  }

  @Override
  public void buildOnce(SensorContext context) {
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    mainFiles.forEach(f ->  inputFileToFQN.put(f, SymbolUtils.fullyQualifiedModuleName(packageName(f), f.filename())));
    LOG.debug("Input files for indexing: " + mainFiles);
    if (shouldOptimizeAnalysis(context)) {
      computeGlobalSymbolsUsingCache(context);
      return;
    }
    PerformanceMeasure.Duration duration = PerformanceMeasure.start("ProjectLevelSymbolTable");
    computeGlobalSymbols(mainFiles, context);
    duration.stop();
  }

  private boolean shouldOptimizeAnalysis(SensorContext context) {
    return caching.isCacheEnabled()
      && (context.canSkipUnchangedFiles() || context.config().getBoolean(SONAR_CAN_SKIP_UNCHANGED_FILES_KEY).orElse(false));
  }

  private void computeGlobalSymbolsUsingCache(SensorContext context) {
    LOG.info("Using cached data to retrieve global symbols.");
    // TODO SONARPY-1196: Compute deleted files here through diff with what was saved and include them to projectModuleFQNs / modifiedFiles
    Set<String> projectModulesFQNs = new HashSet<>(inputFileToFQN.values());
    Map<String, Set<String>> importsByModule = new HashMap<>();
    List<InputFile> impactfulFiles = new ArrayList<>();
    List<String> impactfulFilesFQN = new ArrayList<>();
    for (InputFile inputFile : mainFiles) {
      String currFQN = inputFileToFQN.get(inputFile);
      boolean isUnimpacted = tryToUseCache(importsByModule, inputFile, currFQN);
      if (!isUnimpacted) {
        // Failed to retrieve some data: consider the file as impactful.
        impactfulFiles.add(inputFile);
        impactfulFilesFQN.add(currFQN);
      }
    }
    Set<String> impactedModulesFQN = DependencyGraph.from(importsByModule, projectModulesFQNs).impactedModules(impactfulFilesFQN);
    mainFiles.stream().filter(f -> !impactedModulesFQN.contains(inputFileToFQN.get(f))).forEach(skippableFiles::add);
    // No project level information is stored for test files. It is therefore impossible for a change in a test file to impact other files.
    testFiles.stream().filter(f -> f.status().equals(InputFile.Status.SAME)).forEach(skippableFiles::add);
    LOG.info(
      "Cached information of global symbols will be used for {} out of {} main files. Global symbols will be recomputed for the remaining files.",
      mainFiles.size() - impactfulFiles.size(),
      mainFiles.size()
    );
    LOG.info("Optimized analysis can be performed for {} out of {} files.", skippableFiles.size(), mainFiles.size() + testFiles.size());
    // Although we need to analyze all impacted files, we only need to recompute global symbols for modified files (no cross-file dependencies in the project symbol table)
    computeGlobalSymbols(impactfulFiles, context);
  }

  private boolean tryToUseCache(Map<String, Set<String>> importsByModule, InputFile inputFile, String currFQN) {
    if (!inputFile.status().equals(InputFile.Status.SAME)) {
      return false;
    }

    Set<String> imports = caching.readImportMapEntry(inputFile.key());
    if (imports != null) {
      importsByModule.put(currFQN, imports);
    }
    Set<Descriptor> descriptors = caching.readProjectLevelSymbolTableEntry(inputFile.key());
    if (descriptors != null && imports != null) {
      saveRetrievedDescriptors(inputFile.key(), descriptors, caching);
      return true;
    }

    return false;
  }

  private void saveRetrievedDescriptors(String fileKey, Set<Descriptor> descriptors, Caching caching) {
    projectLevelSymbolTable().insertEntry(fileKey, descriptors);
    caching.copyFromPrevious(fileKey);
  }

  public void computeGlobalSymbols(List<InputFile> files, SensorContext context) {
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
    if (caching.isCacheEnabled()) {
      saveGlobalSymbolsInCache(files);
    }
  }

  private void saveGlobalSymbolsInCache(List<InputFile> files) {
    for (InputFile inputFile : files) {
      String moduleFQN = inputFileToFQN.get(inputFile);
      Set<Descriptor> descriptors = projectLevelSymbolTable().descriptorsForModule(moduleFQN);
      caching.writeProjectLevelSymbolTableEntry(inputFile.key(), descriptors);
      caching.writeImportsMapEntry(inputFile.key(), projectLevelSymbolTable().importsByModule().get(moduleFQN));
    }
  }

  @Override
  public boolean canBeScannedWithoutParsing(InputFile inputFile) {
    return skippableFiles.contains(inputFile);
  }

  @Override
  public CacheContext cacheContext() {
    return caching.cacheContext();
  }
}
