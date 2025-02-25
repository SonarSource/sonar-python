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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatcher;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.plugins.python.SensorTelemetryStorage;
import org.sonar.plugins.python.TelemetryMetricKey;

import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;

class DependencyTelemetryTest {
  private SensorContextTester context;
  private SensorTelemetryStorage sensorTelemetryStorage;

  @BeforeEach
  void setUp() {
    context = spy(SensorContextTester.create(Path.of(".")));
    sensorTelemetryStorage = new SensorTelemetryStorage();
  }

  @Test
  void test() {
    addRequirementsTxtFile();
    addPyProjectTomlFile();

    new DependencyTelemetry(sensorTelemetryStorage, context.fileSystem()).process();

    sensorTelemetryStorage.send(context);

    verify(context).addTelemetryProperty(
      eq(TelemetryMetricKey.PYTHON_DEPENDENCIES.key()),
      argThat(dependenciesMatch(
        "package1",
        "package2",
        "subdir-package1",
        "pyproject-package1",
        "pyproject-package2",
        "subdir-pyproject-package1"
        )));
    verify(context).addTelemetryProperty(TelemetryMetricKey.PYTHON_DEPENDENCIES_FORMAT_VERSION.key(), "1");
  }


  private void addRequirementsTxtFile() {
    addFileToContext(Path.of("./requirements.txt"), """
      package1
      package2
      """);
    addFileToContext(Path.of("./subdir/requirements.txt"), """
      subdir-package1
      """);
  }

  private void addPyProjectTomlFile() {
    addFileToContext(Path.of("./pyproject.toml"), """
      [project]
      dependencies = [
        "pyproject_package1",
        "pyproject-package2",
      ]
      """);
    addFileToContext(Path.of("./subdir/pyproject.toml"), """
      [project]
      dependencies = [
        "subdir.pyproject-package1",
        "package-to-be-removed-by-post-processor-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      ]
      """);
  }

  private void addFileToContext(Path path, String content) {
    InputFile inputFile = new TestInputFileBuilder("moduleKey", path.toString())
      .setModuleBaseDir(path.getParent())
      .setCharset(StandardCharsets.UTF_8)
      .setContents(content)
      .build();

    context.fileSystem().add(inputFile);
  }

  private ArgumentMatcher<String> dependenciesMatch(String ...packages) {
    Set<String> expectedPackages = Arrays.stream(packages).collect(Collectors.toSet());
    return new ArgumentMatcher<>() {
      @Override
      public boolean matches(String str) {
        Set<String> foundPackages = Arrays.stream(str.split(",")).collect(Collectors.toSet());
        return foundPackages.equals(expectedPackages);
      }

      @Override
      public String toString() {
        return "\"" + String.join(",", expectedPackages) + "\" (in any order)";
      }
    };
  }
}
