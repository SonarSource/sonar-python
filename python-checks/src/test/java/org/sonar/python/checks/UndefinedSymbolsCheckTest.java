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
package org.sonar.python.checks;

import java.util.Arrays;
import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class UndefinedSymbolsCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/undefinedSymbols/undefinedSymbols.py", new UndefinedSymbolsCheck());
  }

  @Test
  public void test_wildcard_import() {
    PythonCheckVerifier.verifyNoIssue(
            Arrays.asList("src/test/resources/checks/undefinedSymbols/withWildcardImport.py","src/test/resources/checks/undefinedSymbols/mod.py" ),
            new UndefinedSymbolsCheck());
  }

  @Test
  public void test_wildcard_import_all_property() {
    PythonCheckVerifier.verifyNoIssue(
            Arrays.asList("src/test/resources/checks/undefinedSymbols/importWithAll.py", "src/test/resources/checks/undefinedSymbols/packageUsingAll/__init__.py"),
            new UndefinedSymbolsCheck());
  }

  @Test
  public void test_unresolved_wildcard_import() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedSymbols/withUnresolvedWildcardImport.py", new UndefinedSymbolsCheck());
  }

  @Test
  public void test_dynamic_globals() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedSymbols/withGlobals.py", new UndefinedSymbolsCheck());
  }

}
