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
package org.sonar.python.checks.utils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.types.v2.ModuleType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PythonMultiFileVerifierTest {

  @Test
  void map() {
    var pathToCode = Map.of("foo.py", "class Foo: ...");

    var result = PythonMultiFileVerifier.mapFiles(pathToCode, "", PythonVisitorContext::moduleType);
    assertThat(result.get("foo.py"))
      .isNotNull()
      .extracting(ModuleType::name)
      .isEqualTo("foo");
  }

  @Test
  void newBaseDirEmpty() {
    var newBaseDir = PythonMultiFileVerifier.getNewBaseDir("", Path.of("temp"));
    assertThat(newBaseDir).isEqualTo("temp");
  }

  @Test
  void newBaseDirNotEmpty() {
    var newBaseDir = PythonMultiFileVerifier.getNewBaseDir("my_dir", Path.of("temp"));
    assertThat(newBaseDir).isEqualTo("temp" + File.separator + "my_dir");
  }

  @Test
  void createTemporaryDirectorException() {
    try (var files = Mockito.mockStatic(Files.class)) {
      files.when(() -> Files.createTempDirectory(""))
        .thenThrow(IOException.class);

      assertThatThrownBy(PythonMultiFileVerifier::createTemporaryDirectory)
        .isInstanceOf(IllegalStateException.class);
    }
  }

  @Test
  void writeTempFileException() {
    var entry = Map.entry("foo.py", "class Foo: ...");
    var tempDirectoryPath = Path.of("temp");
    try (var files = Mockito.mockStatic(Files.class)) {
      files.when(() -> Files.newBufferedWriter(Mockito.any()))
        .thenThrow(IOException.class);

      assertThatThrownBy(() -> PythonMultiFileVerifier.writeTempFile(entry, tempDirectoryPath))
        .isInstanceOf(IllegalStateException.class);
    }
  }

}
