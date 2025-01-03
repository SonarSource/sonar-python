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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class NeedlessPassCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/needlessPass.py", new NeedlessPassCheck());
  }

  @Test
  void quick_fix_test() {
    var expected = "def my_method():\n" +
      "    print('foo')\n" +
      "    print('foo')\n";

    var input = "def my_method():\n" +
      "    print('foo')\n" +
      "    print('foo')\n" +
      "    pass\n";
    PythonQuickFixVerifier.verify(new NeedlessPassCheck(), input, expected);
    PythonQuickFixVerifier.verifyQuickFixMessages(new NeedlessPassCheck(), input, NeedlessPassCheck.QUICK_FIX_MESSAGE);

    input = "def my_method():\n" +
      "    print('foo')\n" +
      "    pass\n" +
      "    print('foo')\n";
    PythonQuickFixVerifier.verify(new NeedlessPassCheck(), input, expected);


    expected = "def my_method():\n" +
      "    print('foo'); print('foo')\n";

    input = "def my_method():\n" +
      "    print('foo'); pass; print('foo')\n";
    PythonQuickFixVerifier.verify(new NeedlessPassCheck(), input, expected);

    input = "def my_method():\n" +
      "    print('foo'); print('foo'); pass\n";
    PythonQuickFixVerifier.verify(new NeedlessPassCheck(), input, expected);
  }

}
