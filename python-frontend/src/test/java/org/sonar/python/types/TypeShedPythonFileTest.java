/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.types;

import java.io.InputStream;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;

import static org.assertj.core.api.Assertions.assertThat;


public class TypeShedPythonFileTest {

  @org.junit.Rule
  public LogTester logTester = new LogTester();

  @Test
  public void existing_resource() {
    TypeShedPythonFile typeShedPythonFile = new TypeShedPythonFile(TypeShedPythonFileTest.class.getResourceAsStream("/typeshed.pyi"), "");
    assertThat(typeShedPythonFile.uri()).isNull();
    assertThat(typeShedPythonFile.fileName()).isEmpty();
    assertThat(typeShedPythonFile.content()).isEqualTo("'hello'");
  }

  @Test
  public void error_while_reading_resource() {
    TypeShedPythonFile typeShedPythonFile = new TypeShedPythonFile(Mockito.mock(InputStream.class), "");
    assertThat(typeShedPythonFile.uri()).isNull();
    assertThat(typeShedPythonFile.fileName()).isEmpty();
    assertThat(typeShedPythonFile.content()).isEqualTo("");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Unable to read builtin types.");
  }
}
