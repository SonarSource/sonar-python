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
package org.sonar.python.checks;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UndefinedSymbolsCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/undefinedSymbols/undefinedSymbols.py", new UndefinedSymbolsCheck());
  }

  @Test
  void test_wildcard_import() {
    PythonCheckVerifier.verifyNoIssue(
            Arrays.asList("src/test/resources/checks/undefinedSymbols/withWildcardImport.py","src/test/resources/checks/undefinedSymbols/mod.py" ),
            new UndefinedSymbolsCheck());
  }

  @Test
  void test_wildcard_import_all_property() {
    PythonCheckVerifier.verifyNoIssue(
            Arrays.asList("src/test/resources/checks/undefinedSymbols/importWithAll.py", "src/test/resources/checks/undefinedSymbols/packageUsingAll/__init__.py"),
            new UndefinedSymbolsCheck());
  }

  @Test
  void test_unresolved_wildcard_import() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedSymbols/withUnresolvedWildcardImport.py", new UndefinedSymbolsCheck());
  }

  @Test
  void test_dynamic_globals() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedSymbols/withGlobals.py", new UndefinedSymbolsCheck());
  }

}
