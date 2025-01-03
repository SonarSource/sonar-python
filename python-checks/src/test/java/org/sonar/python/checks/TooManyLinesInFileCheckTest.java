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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class TooManyLinesInFileCheckTest {

  private TooManyLinesInFileCheck check = new TooManyLinesInFileCheck();

  @Test
  void test_negative() {
    check.maximum = 3;
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tooManyLinesInFile-3.py", check);
  }

  @Test
  void test_positive() {
    check.maximum = 2;
    PythonCheckVerifier.verify("src/test/resources/checks/tooManyLinesInFile-2.py", check);
  }

  @Test
  void test_empty_file() {
    check.maximum = 2;
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tooManyLinesInFile-empty.py", check);
  }

}
