/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.parser;

import java.io.File;
import java.nio.file.Files;
import java.util.Collection;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import static java.nio.charset.StandardCharsets.UTF_8;

class PythonParserTest {

  private final PythonParser parser = PythonParser.create();

  @Test
  void test() throws Exception {
    Collection<File> files = listFiles();
    for (File file : files) {
      String fileContent = new String(Files.readAllBytes(file.toPath()), UTF_8);
      parser.parse(fileContent);
    }
  }

  private static Collection<File> listFiles() {
    File dir = new File("src/test/resources/parser/");
    return FileUtils.listFiles(dir, new String[]{"py"}, true);
  }

}
