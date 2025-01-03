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
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class InstanceAndClassMethodsAtLeastOnePositionalCheckTest {

  private final PythonCheck check = new InstanceAndClassMethodsAtLeastOnePositionalCheck();
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/instanceAndClassMethodAtLeastOnePositional.py", check);
  }

  @Test
  void class_method_quickfix() {
    String codeWithIssue = "" +
      "class Foo():\n" +
      "    @classmethod\n" +
      "    def bar(): pass";

    String fixedCodeWithClassParameter = "" +
      "class Foo():\n" +
      "    @classmethod\n" +
      "    def bar(cls): pass";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCodeWithClassParameter);
  }

  @Test
  void regular_method_quickfix() {
    String codeWithIssue = "" +
      "class Foo():\n" +
      "    def bar(): pass";

    String fixedCodeWithSelfParameter = "" +
      "class Foo():\n" +
      "    def bar(cls): pass";

    String fixedCodeWithClassParameter = "" +
      "class Foo():\n" +
      "    def bar(self): pass";

    PythonQuickFixVerifier.verify(check, codeWithIssue,
      fixedCodeWithSelfParameter,
      fixedCodeWithClassParameter);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue,
      "Add 'cls' as the first parameter.",
      "Add 'self' as the first parameter.");
  }

  @Test
  void no_pos_args_quickfix() {
    String codeWithIssue = "" +
      "class Foo():\n" +
      "    @classmethod\n" +
      "    def bar(*, param): pass";

    String fixedCodeWithClassParameter = "" +
      "class Foo():\n" +
      "    @classmethod\n" +
      "    def bar(cls, *, param): pass";

    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCodeWithClassParameter);
  }
}
