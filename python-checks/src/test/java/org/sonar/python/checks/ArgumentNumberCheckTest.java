/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

class ArgumentNumberCheckTest {

  @Test
  void library_overloads() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentNumberLibraryOverloads.py", new ArgumentNumberCheck());
  }

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentNumber.py", new ArgumentNumberCheck());
  }

  @Test
  void dependency_versions() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentNumberDependencyVersions.py", new ArgumentNumberCheck());
  }

  @Test
  void bound_callable_alias() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentNumberCallableAlias.py", new ArgumentNumberCheck());
  }

  @Test
  void test_multiple_files() {
    PythonCheckVerifier.verify(
      Arrays.asList("src/test/resources/checks/argumentNumberWithImport.py", "src/test/resources/checks/argumentNumberImported.py"),
      new ArgumentNumberCheck());
  }

  @Test
  void decorated_super_calls_with_incomplete_ancestry_across_files() {
    PythonCheckVerifier.verify(Arrays.asList(
      "src/test/resources/checks/argumentNumberIncompleteSuper/usage.py",
      "src/test/resources/checks/argumentNumberIncompleteSuper/base.py",
      "src/test/resources/checks/argumentNumberIncompleteSuper/decorators.py"), new ArgumentNumberCheck());
  }
}
