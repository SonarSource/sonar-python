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
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
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
import org.sonar.plugins.python.api.PythonCustomRuleRepositoryWrapper;
import org.sonar.plugins.python.api.PythonFileConsumer;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.SonarLintCacheWrapper;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.architecture.ArchitectureCallbackWrapper;
import org.sonar.plugins.python.editions.RepositoryInfoProvider;
import org.sonar.plugins.python.editions.RepositoryInfoProvider.RepositoryInfo;
import org.sonar.plugins.python.editions.RepositoryInfoProviderWrapper;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.indexer.PythonIndexerWrapper;
import org.sonar.plugins.python.indexer.SonarQubePythonIndexer;
import org.sonar.plugins.python.nosonar.NoSonarLineInfoCollector;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.types.TypeShed;
import org.sonarsource.performance.measure.PerformanceMeasure;

import static org.sonar.plugins.python.PythonScanner.THREADS_PROPERTY_NAME;
import static org.sonar.plugins.python.Scanner.PARALLEL_PROPERTY_NAME;
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
  private final PythonFileConsumer architectureCallback;

  private final SonarLintCache sonarLintCache;
  private final AnalysisWarningsWrapper analysisWarnings;
  private static final Logger LOG = LoggerFactory.getLogger(PythonSensor.class);
  static final String UNSET_VERSION_WARNING = """
    Your code is analyzed as compatible with all Python 3 versions by default. \
    You can get a more precise analysis by setting the exact Python version in your configuration via the parameter "sonar.python.version"\
    """;

  private final SensorTelemetryStorage sensorTelemetryStorage;
  private final NoSonarLineInfoCollector noSonarLineInfoCollector;
  private final ProjectConfigurationBuilder projectConfigurationBuilder;

  public PythonSensor(
    FileLinesContextFactory fileLinesContextFactory,
    CheckFactory checkFactory,
    NoSonarFilter noSonarFilter,
    PythonCustomRuleRepositoryWrapper customRuleRepositoriesWrapper,
    PythonIndexerWrapper indexerWrapper,
    SonarLintCacheWrapper sonarLintCacheWrapper,
    AnalysisWarningsWrapper analysisWarnings,
    RepositoryInfoProviderWrapper editionMetadataProviderWrapper,
    ArchitectureCallbackWrapper architectureUDGBuilderWrapper,
    NoSonarLineInfoCollector noSonarLineInfoCollector,
    ProjectConfigurationBuilder projectConfigurationBuilder) {
    this.noSonarLineInfoCollector = noSonarLineInfoCollector;
    this.projectConfigurationBuilder = projectConfigurationBuilder;

    this.checks = createPythonChecks(checkFactory, editionMetadataProviderWrapper.infoProviders())
        .addCustomChecks(customRuleRepositoriesWrapper.customRuleRepositories());
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.indexer = indexerWrapper.indexer();
    this.sonarLintCache = sonarLintCacheWrapper.sonarLintCache();
    this.analysisWarnings = analysisWarnings;
    this.sensorTelemetryStorage = new SensorTelemetryStorage();
    this.architectureCallback = architectureUDGBuilderWrapper.architectureUdgBuilder();
  }

  private static PythonChecks createPythonChecks(CheckFactory checkFactory, RepositoryInfoProvider[] editionMetadataProviders) {
    PythonChecks checks = new PythonChecks(checkFactory);
    for (RepositoryInfoProvider repositoryInfoProvider : editionMetadataProviders) {
      RepositoryInfo repositoryInfo = repositoryInfoProvider.getInfo();
      checks.addChecks(repositoryInfo.repositoryKey(), repositoryInfo.checks());
    }
    return checks;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(Python.KEY)
      .name("Python Sensor");
  }

  @Override
  public void execute(SensorContext context) {
    Instant sensorStartTime = Instant.now();
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
    PythonIndexer pythonIndexer = this.indexer != null ? this.indexer : new SonarQubePythonIndexer(pythonFiles, cacheContext, context, projectConfigurationBuilder);
    pythonIndexer.setSonarLintCache(sonarLintCache);
    TypeShed.setProjectLevelSymbolTable(pythonIndexer.projectLevelSymbolTable());
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, PythonParser::create,
      pythonIndexer, architectureCallback, noSonarLineInfoCollector);
    scanner.execute(pythonFiles, context);
    Duration sensorTime = Duration.between(sensorStartTime, Instant.now());

    updateDatabricksTelemetry(scanner);
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOSONAR_RULE_ID_KEYS, noSonarLineInfoCollector.getSuppressedRuleIds());
    updatePerformanceTelemetry(context, pythonFiles, scanner, sensorTime);
    sensorTelemetryStorage.send(context);
    durationReport.stop();
  }

  private void updatePerformanceTelemetry(SensorContext context, List<PythonInputFile> pythonFiles, PythonScanner scanner, Duration sensorTime) {
    context.config().getInt(THREADS_PROPERTY_NAME).ifPresent(v -> sensorTelemetryStorage.updateMetric(TelemetryMetricKey.ANALYSIS_THREADS_PARAM_KEY, v));
    context.config().getBoolean(PARALLEL_PROPERTY_NAME).ifPresent(v -> sensorTelemetryStorage.updateMetric(TelemetryMetricKey.PARALLEL_ANALYSIS_KEY, v));
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.PYTHON_NUMBER_OF_FILES_KEY, pythonFiles.size());
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.ANALYSIS_THREADS_KEY, scanner.getNumberOfThreads(context));
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.ANALYSIS_DURATION_KEY, sensorTime.toSeconds());
  }

  private static List<PythonInputFile> getInputFiles(SensorContext context) {
    FilePredicates p = context.fileSystem().predicates();
    Iterable<InputFile> it = context.fileSystem().inputFiles(p.and(p.hasLanguage(Python.KEY)));
    List<PythonInputFile> list = new ArrayList<>();
    it.forEach(f -> list.add(new PythonInputFileImpl(f)));
    return Collections.unmodifiableList(list);
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
