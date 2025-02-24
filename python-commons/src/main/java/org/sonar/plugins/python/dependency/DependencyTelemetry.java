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
package org.sonar.plugins.python.dependency;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.api.batch.fs.FileSystem;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.SensorTelemetryStorage;
import org.sonar.plugins.python.TelemetryMetricKey;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

public class DependencyTelemetry {
  private final SensorTelemetryStorage sensorTelemetryStorage;
  private final FileSystem fileSystem;

  public DependencyTelemetry(SensorTelemetryStorage sensorTelemetryStorage, FileSystem fileSystem) {
    this.sensorTelemetryStorage = sensorTelemetryStorage;
    this.fileSystem = fileSystem;
  }
  
  public void process() {
    Dependencies dependencies = collectDependencies();
    var postProcessedDependencies =  DependencyPostProcessor.process(dependencies);
    sendDependencies(postProcessedDependencies);
  }

  private Dependencies collectDependencies() {
    return Stream.concat(
      collectRequirementsTxtDependencies(),
      collectPyProjectTomlDependencies()
    ).collect(Dependencies.mergeCollector());
  }

  private Stream<Dependencies> collectRequirementsTxtDependencies() {
    List<InputFile> requirementsTxtFiles = collectInputFiles("requirements.txt");
    return requirementsTxtFiles.stream().map(RequirementsTxtParser::parseRequirementFile);
  }

  private Stream<Dependencies> collectPyProjectTomlDependencies() {
    List<InputFile> requirementsTxtFiles = collectInputFiles("pyproject.toml");
    return requirementsTxtFiles.stream().map(PyProjectTomlParser::parse);
  }

  private List<InputFile> collectInputFiles(String fileName) {
    var list = new ArrayList<InputFile>();
    fileSystem.inputFiles(fileSystem.predicates().hasFilename(fileName)).forEach(list::add);
    return list;
  }

  private void sendDependencies(Dependencies dependencies) {
    String dependenciesString = formatDependencies(dependencies);
    sensorTelemetryStorage.updateMetric(TelemetryMetricKey.PYTHON_DEPENDENCIES, dependenciesString);
  }

  private static String formatDependencies(Dependencies dependencies) {
    return dependencies.dependencies().stream()
      .map(Dependency::name)
      .collect(Collectors.joining(","));
  }
}
