/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
import org.sonar.python.PythonFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class SonarQubePythonFileTest {

  private InputFile inputFile = mock(InputFile.class);

  @Rule
  public ExpectedException thrown = ExpectedException.none();

  @Test
  public void known_file() throws Exception {
    when(inputFile.contents()).thenReturn("Hello 6.2!");
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile);
    assertThat(PythonFile.content()).isEqualTo("Hello 6.2!");
  }

  @Test
  public void unknown_file() throws Exception {
    when(inputFile.contents()).thenThrow(new FileNotFoundException());
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile);
    thrown.expect(IllegalStateException.class);
    PythonFile.content();
  }

}
