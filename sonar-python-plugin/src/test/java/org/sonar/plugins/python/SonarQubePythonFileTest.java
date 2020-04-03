/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.FileNotFoundException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVersion;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class SonarQubePythonFileTest {

  private InputFile inputFile = mock(InputFile.class, "file1.py");

  @Rule
  public ExpectedException thrown = ExpectedException.none();

  @Test
  public void known_file() throws Exception {
    when(inputFile.contents()).thenReturn("Hello 6.2!");
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile, PythonVersion.allVersions());
    assertThat(pythonFile.content()).isEqualTo("Hello 6.2!");
    assertThat(pythonFile.toString()).isEqualTo(inputFile.toString());
    assertThat(pythonFile.uri()).isEqualTo(inputFile.uri());
  }

  @Test
  public void python_version() {
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile, PythonVersion.fromString("> 3"));
    assertThat(pythonFile.pythonVersion().isPython3Only()).isTrue();
  }

  @Test
  public void unknown_file() throws Exception {
    when(inputFile.contents()).thenThrow(new FileNotFoundException());
    PythonFile pythonFile = SonarQubePythonFile.create(inputFile, PythonVersion.allVersions());
    thrown.expect(IllegalStateException.class);
    pythonFile.content();
  }

}
