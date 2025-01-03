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
package org.sonar.python.checks;

import java.util.Arrays;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ArgumentNumberCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentNumber.py", new ArgumentNumberCheck());
  }

  @Test
  void test_multiple_files() {
    PythonCheckVerifier.verify(
      Arrays.asList("src/test/resources/checks/argumentNumberWithImport.py", "src/test/resources/checks/argumentNumberImported.py"),
      new ArgumentNumberCheck());
  }
}
