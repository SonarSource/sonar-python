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
package org.sonar.python.checks.utils;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
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
    var symbolTable = TestPythonVisitorRunner.globalSymbols(pathToCode, baseDir);
    Map<String, R> results = new HashMap<>();

    pathToCode.forEach((path, code) -> {
      var mockFile = new TestPythonVisitorRunner.MockPythonFile(baseDir, path, code);
      var context = TestPythonVisitorRunner.createContext(mockFile, new File(baseDir), packageName, symbolTable, CacheContextImpl.dummyCache());
      results.put(path, function.apply(context));
    });

    return results;
  }

}
