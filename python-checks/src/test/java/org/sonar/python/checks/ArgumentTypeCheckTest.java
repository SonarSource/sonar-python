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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ArgumentTypeCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentType.py", new ArgumentTypeCheck());
  }

  @Test
  void test_overloaded_functions() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentType_overloaded_functions.py", new ArgumentTypeCheck());
  }

  @Test
  void importedTypesTest() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/argumentTypeImporting.py",
        "src/test/resources/checks/argumentTypeImported.py"
      ),
      new ArgumentTypeCheck()
    );
  }
}
