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

import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

public class CustomStubSanityTest {

  @Test
  public void test() throws IOException, URISyntaxException {
    URL customStubsURL = CustomStubSanityTest.class.getResource("/org/sonar/python/types/custom");
    Path customStubsPath = Paths.get(customStubsURL.toURI());

    Set<String> moduleNames = Files.find(customStubsPath, Integer.MAX_VALUE, ((path, basicFileAttributes) -> basicFileAttributes.isRegularFile()))
      .map(customStubsPath::relativize)
      .map(CustomStubSanityTest::pathToModuleName)
      .collect(Collectors.toSet());

    assertThat(moduleNames).isNotEmpty();
    for (String module : moduleNames) {
      // Make sure that each module we describe with a custom stub declares something and has a valid syntax.
      try {
        Set<Symbol> symbols = TypeShed.symbolsForModule(module);
        assertThat(symbols).isNotEmpty();
      } catch (RecognitionException ex) {
        fail(String.format("Syntax error in stub file while resolving module '%s' (error may be in a referenced module).", module), ex);
      }
    }
  }

  private static String pathToModuleName(Path path) {
    String pathStr = path.toString();
    if (path.getFileName().toString().equals("__init__.pyi")) {
      // If we have an __init__.py file, use the parent directory name for the module
      pathStr = path.getParent().toString();
    } else {
      // Otherwise use the file name, but drop the extension
      pathStr = pathStr.substring(0, pathStr.lastIndexOf(".pyi"));
    }

    return pathStr.replace(File.separator, ".");
  }
}
