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
package org.sonar.python.semantic;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class SymbolUtilsTest {

  @Test
  public void package_name_by_file() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/__init__.py"), baseDir)).isEqualTo("sound");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/__init__.py"), baseDir)).isEqualTo("sound.formats");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/wavread.py"), baseDir)).isEqualTo("sound.formats");
  }

  @Test
  public void fqn_by_package_with_subpackage() {
    assertThat(SymbolUtils.fullyQualifiedModuleName("", "foo.py")).isEqualTo("foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "__init__.py")).isEqualTo("foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "foo.py")).isEqualTo("foo.foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("foo", "foo")).isEqualTo("foo.foo");
    assertThat(SymbolUtils.fullyQualifiedModuleName("curses", "ascii.py")).isEqualTo("curses.ascii");
  }

  @Test
  public void path_of() throws IOException, URISyntaxException {
    PythonFile pythonFile = Mockito.mock(PythonFile.class);
    URI uri = Files.createTempFile("foo.py", "py").toUri();
    Mockito.when(pythonFile.uri()).thenReturn(uri);
    assertThat(pathOf(pythonFile)).isEqualTo(Paths.get(uri));

    uri = new URI("myscheme", null, "/file1.py", null);

    Mockito.when(pythonFile.uri()).thenReturn(uri);
    assertThat(pathOf(pythonFile)).isNull();

    Mockito.when(pythonFile.uri()).thenThrow(InvalidPathException.class);
    assertThat(pathOf(pythonFile)).isNull();
  }
}
