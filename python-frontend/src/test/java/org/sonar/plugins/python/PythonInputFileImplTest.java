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

import java.io.IOException;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.plugins.python.PythonInputFile.Kind;

import static org.assertj.core.api.Assertions.assertThat;

class PythonInputFileImplTest {

  @Test
  void shouldHavePythonKind() {
    PythonInputFile inputFile = new PythonInputFileImpl(createWrappedFile());
    assertThat(inputFile.kind()).isEqualTo(Kind.PYTHON);
  }

  @Test
  void shouldHaveTheWrappedFileToString() {
    InputFile wrappedFile = createWrappedFile();
    PythonInputFile inputFile = new PythonInputFileImpl(wrappedFile);
    assertThat(inputFile).hasToString(wrappedFile.toString());
  }

  @Test
  void shouldReturnTheContentOfTheWrappedFile() throws IOException {
    InputFile wrappedFile = createWrappedFile();
    PythonInputFile inputFile = new PythonInputFileImpl(wrappedFile);
    assertThat(inputFile.contents()).isEqualTo(wrappedFile.contents());
  }

  private InputFile createWrappedFile() {
    return TestInputFileBuilder.create("moduleKey", "name").setContents("Test").build();
  }
}
