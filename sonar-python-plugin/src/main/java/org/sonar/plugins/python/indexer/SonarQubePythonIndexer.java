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
import javax.annotation.Nullable;
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

  private final List<InputFile> mainFiles;
  private final List<InputFile> testFiles;
  private final Set<InputFile> skippableFiles;
  @Nullable
  private final Caching caching;
  private final Map<InputFile, String> inputFileToFQN;
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
    this.inputFileToFQN = new HashMap<>();
    inputFiles.forEach(f -> {
      if (f.type().equals(InputFile.Type.MAIN)) {
        mainFiles.add(f);
        inputFileToFQN.put(f, SymbolUtils.fullyQualifiedModuleName(packageName(f), f.filename()));
      } else {
        testFiles.add(f);
      }
    });
    this.skippableFiles = new HashSet<>();
    this.caching = cacheContext.isCacheEnabled() ? new Caching(cacheContext) : null;
  }

  @Override
  public void buildOnce(SensorContext context) {
    this.projectBaseDirAbsolutePath = context.fileSystem().baseDir().getAbsolutePath();
    LOG.debug("Input files for indexing: " + mainFiles);
    boolean shouldOptimizeAnalysis = context.config().getBoolean(SONAR_CAN_SKIP_UNCHANGED_FILES_KEY).orElse(false) && caching != null;
    if (!shouldOptimizeAnalysis) {
      PerformanceMeasure.Duration duration = PerformanceMeasure.start("ProjectLevelSymbolTable");
      computeGlobalSymbols(mainFiles, context);
      duration.stop();
      return;
    }
    computeGlobalSymbolsUsingCache(context);
  }

  private void computeGlobalSymbolsUsingCache(SensorContext context) {
    LOG.info("Using cached data to retrieve global symbols.");
    // TODO SONARPY-1196: Compute deleted files here through diff with what was saved and include them to projectModuleFQNs / modifiedFiles
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
        saveRetrievedDescriptors(currFQN, descriptors, caching);
      } else {
        // Failed to retrieve some data: consider the file as impactful.
        impactfulFilesFQN.add(currFQN);
        modifiedFiles.add(inputFile);
      }
    }
    Set<String> impactedModulesFQN = DependencyGraph.from(importsByModule, projectModulesFQNs).impactedModules(impactfulFilesFQN);
    mainFiles.stream().filter(f -> !impactedModulesFQN.contains(inputFileToFQN.get(f))).forEach(skippableFiles::add);
    testFiles.stream().filter(f -> f.status().equals(InputFile.Status.SAME)).forEach(skippableFiles::add);
    LOG.info(
      "Cached information of global symbols will be used for {} out of {} main files. Global symbols will be recomputed for the remaining files.",
      mainFiles.size() - impactfulFilesFQN.size(),
      mainFiles.size()
    );
    LOG.info("Optimized analysis can be performed for {} out of {} files.", skippableFiles.size(), mainFiles.size() + testFiles.size());
    // Although we need to analyze all impacted files, we only need to recompute PST entries for modified files (no cross-file dependencies in the project symbol table)
    computeGlobalSymbols(modifiedFiles, context);
  }

  private void saveRetrievedDescriptors(String moduleFqn, Set<Descriptor> descriptors, Caching caching) {
    projectLevelSymbolTable().insertEntry(moduleFqn, descriptors);
    caching.copyFromPrevious(moduleFqn);
  }

  public void computeGlobalSymbols(List<InputFile> files, SensorContext context) {
    GlobalSymbolsScanner globalSymbolsStep = new GlobalSymbolsScanner(context);
    globalSymbolsStep.execute(files, context);
    saveGlobalSymbolsInCache(files);
  }

  private void saveGlobalSymbolsInCache(List<InputFile> files) {
    if (caching != null) {
      for (InputFile inputFile : files) {
        String moduleFQN = inputFileToFQN.get(inputFile);
        Set<Descriptor> descriptors = projectLevelSymbolTable().descriptorsForModule(moduleFQN);
        caching.writeProjectLevelSymbolTableEntry(moduleFQN, descriptors);
        caching.writeImportsMapEntry(moduleFQN, projectLevelSymbolTable().importsByModule().get(moduleFQN));
      }
    }
  }

  @Override
  public boolean canBeScannedWithoutParsing(InputFile inputFile) {
    return skippableFiles.contains(inputFile);
  }
}
