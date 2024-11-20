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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class RandomSeedCheckTest {
  RandomSeedCheck check = new RandomSeedCheck();

  @Test
  void test_numpy() {
    PythonCheckVerifier.verify("src/test/resources/checks/randomSeedNumpy.py", check );
  }
  
  @Test
  void test_sklearn() {
    PythonCheckVerifier.verify("src/test/resources/checks/randomSeedSKlearn.py", check);
  }
}

