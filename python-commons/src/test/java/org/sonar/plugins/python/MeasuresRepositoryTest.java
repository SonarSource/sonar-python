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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MeasuresRepositoryTest {


  @Test
  void isNotebookTest() {
    var regularPythonFile = mock(PythonInputFile.class);
    when(regularPythonFile.kind()).thenReturn(PythonInputFile.Kind.PYTHON);
    assertThat(MeasuresRepository.isNotebook(regularPythonFile)).isFalse();

    var notebookPythonFile = mock(PythonInputFile.class);
    when(notebookPythonFile.kind()).thenReturn(PythonInputFile.Kind.IPYTHON);
    assertThat(MeasuresRepository.isNotebook(notebookPythonFile)).isTrue();
  }
}
