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
package org.sonar.plugins.python;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.SonarProduct;
import org.sonar.api.batch.DependedUpon;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.indexer.SonarQubePythonIndexer;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.checks.CheckList;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.types.TypeShed;
import org.sonarsource.performance.measure.PerformanceMeasure;

import static org.sonar.plugins.python.api.PythonVersionUtils.PYTHON_VERSION_KEY;

@DependedUpon(value = "org.sonar.plugins.python.PythonSensor_before_com.sonarsource.dbd.SonarLintPythonBugDetectionSensor")
public final class PythonSensor implements Sensor {

  private static final String PERFORMANCE_MEASURE_PROPERTY = "sonar.python.performance.measure";
  private static final String PERFORMANCE_MEASURE_FILE_PATH_PROPERTY = "sonar.python.performance.measure.path";
  private static final String PERFORMANCE_MEASURE_DESTINATION_FILE = "sonar-python-performance-measure.json";

  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;
  private final PythonIndexer indexer;

  private final SonarLintCache sonarLintCache;
  private final AnalysisWarningsWrapper analysisWarnings;
  private static final Logger LOG = LoggerFactory.getLogger(PythonSensor.class);
  static final String UNSET_VERSION_WARNING = "Your code is analyzed as compatible with all Python 3 versions by default." +
    " You can get a more precise analysis by setting the exact Python version in your configuration via the parameter \"sonar.python" +
    ".version\"";

  private final SensorTelemetryStorage sensorTelemetryStorage;

  /**
   * Constructor to be used by pico if neither PythonCustomRuleRepository nor PythonIndexer are to be found and injected.
   */
  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory,
    NoSonarFilter noSonarFilter, AnalysisWarningsWrapper analysisWarnings) {
    this(fileLinesContextFactory, checkFactory, noSonarFilter, null, null, null, analysisWarnings);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
    PythonCustomRuleRepository[] customRuleRepositories, AnalysisWarningsWrapper analysisWarnings) {
    this(fileLinesContextFactory, checkFactory, noSonarFilter, customRuleRepositories, null, null, analysisWarnings);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
    PythonIndexer indexer, SonarLintCache sonarLintCache, AnalysisWarningsWrapper analysisWarnings) {
    // ^^ This constructor implicitly assumes that a PythonIndexer and a SonarLintCache are always available at the same time.
    // In practice, this is currently the case, since both are provided by PythonPlugin under the same conditions.
    // See also PythonPlugin::SonarLintPluginAPIManager::addSonarlintPythonIndexer.
    this(fileLinesContextFactory, checkFactory, noSonarFilter, null, indexer, sonarLintCache, analysisWarnings);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
    @Nullable PythonCustomRuleRepository[] customRuleRepositories, @Nullable PythonIndexer indexer,
    @Nullable SonarLintCache sonarLintCache, AnalysisWarningsWrapper analysisWarnings) {
    this.checks = new PythonChecks(checkFactory)
      .addChecks(CheckList.REPOSITORY_KEY, CheckList.getChecks())
      .addCustomChecks(customRuleRepositories);
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.indexer = indexer;
    this.sonarLintCache = sonarLintCache;
    this.analysisWarnings = analysisWarnings;
    this.sensorTelemetryStorage = new SensorTelemetryStorage();
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(Python.KEY)
      .name("Python Sensor");
  }

  @Override
  public void execute(SensorContext context) {
    PerformanceMeasure.Duration durationReport = createPerformanceMeasureReport(context);
    List<PythonInputFile> pythonFiles = getInputFiles(context);
    String[] pythonVersionParameter = context.config().getStringArray(PYTHON_VERSION_KEY);
    if (pythonVersionParameter.length == 0 && context.runtime().getProduct() != SonarProduct.SONARLINT) {
      LOG.warn(UNSET_VERSION_WARNING);
      analysisWarnings.addUnique(UNSET_VERSION_WARNING);
    }
    if (pythonVersionParameter.length != 0) {
      ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.fromStringArray(pythonVersionParameter));
    }
    updatePythonVersionTelemetry(context, pythonVersionParameter);
    CacheContext cacheContext = CacheContextImpl.of(context);
    PythonIndexer pythonIndexer = this.indexer != null ? this.indexer : new SonarQubePythonIndexer(pythonFiles, cacheContext, context);
    pythonIndexer.setSonarLintCache(sonarLintCache);
    TypeShed.setProjectLevelSymbolTable(pythonIndexer.projectLevelSymbolTable());
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, PythonParser.create(),
      pythonIndexer);
    scanner.execute(pythonFiles, context);
    updateDatabricksTelemetry(scanner);
    sensorTelemetryStorage.send(context);
    durationReport.stop();
  }

  private void updateDatabricksTelemetry(PythonScanner scanner) {
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.PYTHON_DATABRICKS_FOUND, scanner.getFoundDatabricks());
  }

  private void updatePythonVersionTelemetry(SensorContext context, String[] pythonVersionParameter) {
    if (context.runtime().getProduct() == SonarProduct.SONARLINT) {
      return;
    }
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.PYTHON_VERSION_SET_KEY, pythonVersionParameter.length != 0);
    if (pythonVersionParameter.length != 0) {
      sensorTelemetryStorage.updateMetric(TelemetryMetricKey.PYTHON_VERSION_KEY, String.join(",", pythonVersionParameter));
    }
  }

  private static List<PythonInputFile> getInputFiles(SensorContext context) {
    FilePredicates p = context.fileSystem().predicates();
    Iterable<InputFile> it = context.fileSystem().inputFiles(p.and(p.hasLanguage(Python.KEY)));
    List<PythonInputFile> list = new ArrayList<>();
    it.forEach(f -> list.add(new PythonInputFileImpl(f)));
    return Collections.unmodifiableList(list);
  }

  private static PerformanceMeasure.Duration createPerformanceMeasureReport(SensorContext context) {
    return PerformanceMeasure.reportBuilder()
      .activate(context.config().getBoolean(PERFORMANCE_MEASURE_PROPERTY).orElse(Boolean.FALSE))
      .toFile(context.config().get(PERFORMANCE_MEASURE_FILE_PATH_PROPERTY)
        .filter(path -> !path.isEmpty())
        .orElseGet(() -> Optional.ofNullable(context.fileSystem().workDir())
          .filter(File::exists)
          .map(file -> file.toPath().resolve(PERFORMANCE_MEASURE_DESTINATION_FILE).toString())
          .orElse(null)))
      .appendMeasurementCost()
      .start("PythonSensor");
  }
}
