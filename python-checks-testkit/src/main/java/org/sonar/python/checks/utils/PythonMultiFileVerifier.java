/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.checks.utils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.caching.CacheContextImpl;

public class PythonMultiFileVerifier {
  private PythonMultiFileVerifier() {
    
  }

  public static <R> Map<String, R> mapFiles(Map<String, String> pathToCode, String baseDir, Function<PythonVisitorContext, R> function) {
    return mapFiles(pathToCode, baseDir, "", function);
  }

  public static <R> Map<String, R> mapFiles(Map<String, String> pathToCode, String baseDir, String packageName, Function<PythonVisitorContext, R> function) {
    var tempDirectoryPath = createTemporaryDirectory();

    var pathToFile = pathToCode.entrySet().stream().map(
      entry -> writeTempFile(entry, tempDirectoryPath)
    ).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    var newBaseDir = getNewBaseDir(baseDir, tempDirectoryPath);

    var symbolTable = TestPythonVisitorRunner.globalSymbols(pathToCode, newBaseDir);
    Map<String, R> results = new HashMap<>();

    pathToFile.forEach((path, file) -> {
      var context = TestPythonVisitorRunner.createContext(file, new File(newBaseDir), packageName, symbolTable, CacheContextImpl.dummyCache());
      results.put(path, function.apply(context));
    });

    return results;

  }

  protected static String getNewBaseDir(String baseDir, Path tempDirectoryPath) {
    return tempDirectoryPath.resolve(baseDir).toString();
  }

  static Path createTemporaryDirectory() {
    try {
      return Files.createTempDirectory("");
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  static Map.Entry<String, File> writeTempFile(Map.Entry<String, String> stringStringEntry, Path tempDirectory) {
    var newPath = tempDirectory.resolve(stringStringEntry.getKey());
    try {
      Files.createDirectories(newPath.getParent());
      var fileWriter = Files.newBufferedWriter(newPath);
      fileWriter.write(stringStringEntry.getValue());
      fileWriter.close();

      return Map.entry(stringStringEntry.getKey(), newPath.toFile());
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

}
