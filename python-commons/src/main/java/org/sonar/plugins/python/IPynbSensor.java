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
package org.sonar.plugins.python;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.api.SonarProduct;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.architecture.DummyArchitectureCallback;
import org.sonar.plugins.python.editions.OpenSourceRepositoryInfoProvider;
import org.sonar.plugins.python.editions.RepositoryInfoProvider;
import org.sonar.plugins.python.editions.RepositoryInfoProvider.RepositoryInfo;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.indexer.SonarQubePythonIndexer;
import org.sonar.plugins.python.nosonar.NoSonarLineInfoCollector;
import org.sonar.plugins.python.telemetry.SensorTelemetryStorage;
import org.sonar.plugins.python.telemetry.TelemetryMetricKey;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.project.config.ProjectConfigurationBuilder;

import static org.sonar.plugins.python.api.PythonVersionUtils.PYTHON_VERSION_KEY;

public final class IPynbSensor implements Sensor {

  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;
  private final PythonIndexer indexer;
  private static final String FAIL_FAST_PROPERTY_NAME = "sonar.internal.analysis.failFast";
  private final SensorTelemetryStorage sensorTelemetryStorage;
  private final NoSonarLineInfoCollector noSonarLineInfoCollector;
  private final ProjectConfigurationBuilder projectConfigurationBuilder;

  public IPynbSensor(FileLinesContextFactory fileLinesContextFactory,
    CheckFactory checkFactory,
    NoSonarFilter noSonarFilter,
    NoSonarLineInfoCollector noSonarLineInfoCollector,
    ProjectConfigurationBuilder projectConfigurationBuilder) {
    this(fileLinesContextFactory,
      checkFactory,
      noSonarFilter,
      null,
      new RepositoryInfoProvider[]{new OpenSourceRepositoryInfoProvider()},
      noSonarLineInfoCollector,
      projectConfigurationBuilder);
  }

  public IPynbSensor(
    FileLinesContextFactory fileLinesContextFactory,
    CheckFactory checkFactory,
    NoSonarFilter noSonarFilter,
    @Nullable PythonIndexer indexer,
    RepositoryInfoProvider[] editionMetadataProviders,
    NoSonarLineInfoCollector noSonarLineInfoCollector,
    ProjectConfigurationBuilder projectConfigurationBuilder) {
    this.noSonarLineInfoCollector = noSonarLineInfoCollector;
    this.projectConfigurationBuilder = projectConfigurationBuilder;

    this.checks = createPythonChecks(checkFactory, editionMetadataProviders);

    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.indexer = indexer;
    this.sensorTelemetryStorage = new SensorTelemetryStorage();
  }

  private static PythonChecks createPythonChecks(CheckFactory checkFactory, RepositoryInfoProvider[] editionMetadataProviders) {
    PythonChecks checks = new PythonChecks(checkFactory);
    for (RepositoryInfoProvider repositoryInfoProvider : editionMetadataProviders) {
      RepositoryInfo repositoryInfo = repositoryInfoProvider.getIPynbInfo();
      checks.addChecks(repositoryInfo.repositoryKey(), repositoryInfo.checks());
    }
    return checks;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(IPynb.KEY)
      .name("IPython Notebooks Sensor");
  }

  @Override
  public void execute(SensorContext context) {
    Instant sensorStartTime = Instant.now();
    List<PythonInputFile> pythonFiles = getInputFiles(context);
    var pythonVersions = context.config().getStringArray(PYTHON_VERSION_KEY);
    if (pythonVersions.length != 0) {
      ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.fromStringArray(pythonVersions));
    }
    if (isInSonarLintRuntime(context)) {
      PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, PythonParser::createIPythonParser,
        indexer, new DummyArchitectureCallback(), noSonarLineInfoCollector);
      scanner.execute(pythonFiles, context);
    } else {
      processNotebooksFiles(pythonFiles, context);
      Duration sensorTime = Duration.between(sensorStartTime, Instant.now());
      sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOTEBOOKS_ANALYSIS_DURATION_KEY, sensorTime.getSeconds());
    }

    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOSONAR_NOTEBOOK_RULE_ID_KEY, noSonarLineInfoCollector.getSuppressedRuleIds());
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOSONAR_NOTEBOOK_COMMENTS_KEY, noSonarLineInfoCollector.getCommentWithExactlyOneRuleSuppressed());

    sensorTelemetryStorage.send(context);
  }

  private void processNotebooksFiles(List<PythonInputFile> pythonFiles, SensorContext context) {
    pythonFiles = parseNotebooks(pythonFiles, context);
    // Disable caching for IPynb files for now see: SONARPY-2020
    CacheContext cacheContext = CacheContextImpl.dummyCache();
    PythonIndexer pythonIndexer = new SonarQubePythonIndexer(pythonFiles, cacheContext, context, projectConfigurationBuilder);
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, PythonParser::createIPythonParser,
      pythonIndexer, new DummyArchitectureCallback(), noSonarLineInfoCollector);
    scanner.execute(pythonFiles, context);
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOTEBOOK_RECOGNITION_ERROR_KEY, scanner.getRecognitionErrorCount());
    updateDatabricksTelemetry(scanner);
  }

  private List<PythonInputFile> parseNotebooks(List<PythonInputFile> pythonFiles, SensorContext context) {
    List<PythonInputFile> generatedIPythonFiles = new ArrayList<>();

    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOTEBOOK_TOTAL_KEY, pythonFiles.size());
    var numberOfExceptions = 0;

    for (PythonInputFile inputFile : pythonFiles) {
      try {
        sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOTEBOOK_PRESENT_KEY, true);
        var result = IpynbNotebookParser.parseNotebook(inputFile);
        result.ifPresent(generatedIPythonFiles::add);
      } catch (Exception e) {
        numberOfExceptions++;
        if (context.config().getBoolean(FAIL_FAST_PROPERTY_NAME).orElse(false) && !isErrorOnTestFile(inputFile)) {
          throw new IllegalStateException("Exception when parsing " + inputFile, e);
        }
      }
    }

    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.NOTEBOOK_EXCEPTION_KEY, numberOfExceptions);
    return generatedIPythonFiles;
  }

  private static boolean isInSonarLintRuntime(SensorContext context) {
    // SL preprocesses notebooks and send us Python files
    // SQ/SC sends us the actual JSON files
    return context.runtime().getProduct().equals(SonarProduct.SONARLINT);
  }

  private static List<PythonInputFile> getInputFiles(SensorContext context) {
    FilePredicates p = context.fileSystem().predicates();
    Iterable<InputFile> it = context.fileSystem().inputFiles(p.and(p.hasLanguage(IPynb.KEY)));
    List<PythonInputFile> list = new ArrayList<>();
    it.forEach(f -> list.add(new PythonInputFileImpl(f)));
    return Collections.unmodifiableList(list);
  }

  private static boolean isErrorOnTestFile(PythonInputFile inputFile) {
    return inputFile.wrappedFile().type() == InputFile.Type.TEST;
  }

  private void updateDatabricksTelemetry(PythonScanner scanner) {
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.IPYNB_DATABRICKS_FOUND, scanner.getFoundDatabricks());
  }

}
