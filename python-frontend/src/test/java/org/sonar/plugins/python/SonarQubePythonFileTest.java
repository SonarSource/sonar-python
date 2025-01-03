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

import java.io.FileNotFoundException;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.PythonFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class SonarQubePythonFileTest {

  private InputFile inputFile = mock(InputFile.class, "file1.py");

  @Test
  void known_file() throws Exception {
    when(inputFile.contents()).thenReturn("Hello 6.2!");
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile);
    assertThat(pythonFile.content()).isEqualTo("Hello 6.2!");
    assertThat(pythonFile).hasToString(inputFile.toString());
    assertThat(pythonFile.uri()).isEqualTo(inputFile.uri());
    assertThat(pythonFile.key()).isEqualTo(inputFile.key());
  }

  @Test
  void unknown_file() throws Exception {
    when(inputFile.contents()).thenThrow(new FileNotFoundException());
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile);
    assertThatThrownBy(pythonFile::content).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void unknown_file_create() throws Exception {
    var pythonInputFile = mock(GeneratedIPythonFile.class);
    when(pythonInputFile.kind()).thenReturn(PythonInputFile.Kind.IPYTHON);
    when(pythonInputFile.contents()).thenThrow(new FileNotFoundException());
    var sqFile = SonarQubePythonFile.create(pythonInputFile);
    assertThatThrownBy(sqFile::content).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void regular_file_create() throws Exception {
    var wrappedFile = mock(InputFile.class);
    when(wrappedFile.contents()).thenReturn("Hello 6.2!");
    var pythonInputFile = mock(PythonInputFile.class);
    when(pythonInputFile.kind()).thenReturn(PythonInputFile.Kind.PYTHON);
    when(pythonInputFile.wrappedFile()).thenReturn(wrappedFile);
    PythonFile pythonFile = SonarQubePythonFile.create(pythonInputFile);
    assertThat(pythonFile.content()).isEqualTo("Hello 6.2!");
  }

  @Test
  void ipynb_file_create() throws Exception {
    var wrappedFile = mock(InputFile.class);
    when(wrappedFile.contents()).thenReturn("Hello wrapped file!");
    var pythonInputFile = mock(GeneratedIPythonFile.class);
    when(pythonInputFile.kind()).thenReturn(PythonInputFile.Kind.IPYTHON);
    when(pythonInputFile.wrappedFile()).thenReturn(wrappedFile);
    when(pythonInputFile.contents()).thenReturn("Hello IPython!");
    PythonFile pythonFile = SonarQubePythonFile.create(pythonInputFile);
    assertThat(pythonFile.content()).isEqualTo("Hello IPython!");
  }
}
