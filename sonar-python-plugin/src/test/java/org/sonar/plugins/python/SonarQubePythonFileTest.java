/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.charset.StandardCharsets;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.utils.Version;
import org.sonar.python.PythonFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class SonarQubePythonFileTest {

  private File baseDir = new File("src/test/resources/org/sonar/plugins/python");
  private File file = new File(baseDir, "SonarQubePythonFile.txt");
  private File unkownFile = new File(baseDir, "xxx");
  private SensorContextTester context = SensorContextTester.create(baseDir);
  private InputFile inputFile = mock(InputFile.class);

  @Rule
  public ExpectedException thrown = ExpectedException.none();

  @Test
  public void sq62() throws Exception {
    setRuntime(Version.create(6, 2));
    when(inputFile.contents()).thenReturn("Hello 6.2!");
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile, context);
    assertThat(PythonFile.content()).isEqualTo("Hello 6.2!");
  }

  @Test
  public void sq62_with_unknown_file() throws Exception {
    setRuntime(Version.create(6, 2));
    when(inputFile.contents()).thenThrow(new FileNotFoundException());
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile, context);
    thrown.expect(IllegalStateException.class);
    PythonFile.content();
  }

  @Test
  public void sq60() throws Exception {
    setRuntime(Version.create(6, 0));
    DefaultFileSystem fs = mock(DefaultFileSystem.class);
    when(fs.encoding()).thenReturn(StandardCharsets.US_ASCII);
    context.setFileSystem(fs);
    when(inputFile.file()).thenReturn(file);
    when(inputFile.charset()).thenReturn(StandardCharsets.UTF_8);
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile, context);
    assertThat(PythonFile.content()).isEqualTo("¡Hello!");
  }

  @Test
  public void sq60_with_unknown_file() throws Exception {
    setRuntime(Version.create(6, 0));
    when(inputFile.file()).thenReturn(unkownFile);
    when(inputFile.charset()).thenReturn(StandardCharsets.UTF_8);
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile, context);
    thrown.expect(IllegalStateException.class);
    PythonFile.content();
  }

  @Test
  public void sq56() {
    setRuntime(Version.create(5, 6));
    DefaultFileSystem fs = mock(DefaultFileSystem.class);
    when(fs.encoding()).thenReturn(StandardCharsets.UTF_8);
    when(inputFile.file()).thenReturn(file);
    context.setFileSystem(fs);
    PythonFile PythonFile = SonarQubePythonFile.create(inputFile, context);
    assertThat(PythonFile.content()).isEqualTo("¡Hello!");
    assertThat(PythonFile.fileName()).isEqualTo(file.getName());
  }

  private void setRuntime(Version version) {
    context.setRuntime(SonarRuntimeImpl.forSonarQube(version, SonarQubeSide.SERVER));
  }

}
