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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class EmptyNestedBlockCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/emptyNestedBlock.py", new EmptyNestedBlockCheck());
  }

  @Test
  void quickFixTest() {
    var check = new EmptyNestedBlockCheck();

    var before = "def foo():\n" +
      "    for i in range(3):\n" +
      "        pass";
    var after = "def foo():\n" +
      "    for i in range(3):\n" +
      "        # TODO: Add implementation\n" +
      "        pass";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, EmptyNestedBlockCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void inlineQuickFixTest() {
    var check = new EmptyNestedBlockCheck();

    var before = "def foo():\n" +
      "    if a < 3: pass\n";

    var after = "def foo():\n" +
      "    if a < 3: \n" +
      "        # TODO: Add implementation\n" +
      "        pass\n";

    PythonQuickFixVerifier.verify(check, before, after);
  }

  @Test
  void rootInlineQuickFixTest() {
    var check = new EmptyNestedBlockCheck();

    var before = "if a < 3: pass\n" +
      "\n" +
      "def foo(a):\n" +
      "  print(a)";

    var after = "if a < 3: \n" +
      "  # TODO: Add implementation\n" +
      "  pass\n" +
      "\n" +
      "def foo(a):\n" +
      "  print(a)";

    PythonQuickFixVerifier.verify(check, before, after);
  }

}
