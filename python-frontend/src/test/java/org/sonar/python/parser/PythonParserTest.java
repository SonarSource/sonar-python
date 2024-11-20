/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.parser;

import java.io.File;
import java.nio.file.Files;
import java.util.Collection;
import org.apache.commons.io.FileUtils;
import org.junit.jupiter.api.Test;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

class PythonParserTest {

  private final PythonParser parser = PythonParser.create();

  private final PythonParser notebookParser = PythonParser.createIPythonParser();

  @Test
  void test_python() throws Exception {
    Collection<File> files = listFiles("python");
    for (File file : files) {
      String fileContent = new String(Files.readAllBytes(file.toPath()), UTF_8);
      assertDoesNotThrow(() -> parser.parse(fileContent));
    }
  }

  private static Collection<File> listFiles(String folderName) {
    File dir = new File(String.format("src/test/resources/parser/%s/", folderName));
    return FileUtils.listFiles(dir, new String[]{"py"}, true);
  }

  @Test
  void test_notebook() throws Exception {
    Collection<File> files = listFiles("notebooks");
    for (File file : files) {
      String fileContent = new String(Files.readAllBytes(file.toPath()), UTF_8);
      assertDoesNotThrow(() -> notebookParser.parse(fileContent));
    }
  }

}
