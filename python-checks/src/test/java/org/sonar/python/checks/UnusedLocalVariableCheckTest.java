/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UnusedLocalVariableCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/unusedLocalVariable.py", new UnusedLocalVariableCheck());
  }

  @Test
  void pandasTest() {
    PythonCheckVerifier.verify("src/test/resources/checks/unusedLocalVariablePandas.py", new UnusedLocalVariableCheck());
  }

  @Test
  void custom() {
    UnusedLocalVariableCheck check = new UnusedLocalVariableCheck();
    check.format = "(_|myignore)";
    PythonCheckVerifier.verify("src/test/resources/checks/unusedLocalVariableCustom.py", check);
  }

  @Test
  void tupleQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def using_tuples():
        x, y = (1, 2)
        print x
      """;
    var after = """
      def using_tuples():
        x, _ = (1, 2)
        print x 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");

    before = """
      def using_tuples():
          x, y = (1, 2)
          y = 5
          print x 
      """;
    after = """
      def using_tuples():
          x, _ = (1, 2)
          y = 5
          print x 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void exceptClauseQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
        def foo():
          try:
            ...
          except Type as e:
            ... 
        """;
    var after = """
      def foo():
        try:
          ...
        except Type:
          ... 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove the unused local variable");
  }

  @Test
  void loopIndexQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def loop_index():
        for i in range(10):
          print("Hello") 
      """;
    var after = """
      def loop_index():
        for _ in range(10):
          print("Hello")
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void loopIndexComprehensionQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def loop_index():
       return [True for i in range(10)]
      """;
    var after = """
      def loop_index():
       return [True for _ in range(10)]
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void loopQuickFixIndexAlreadyTakenTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def a():
          _ = 3
          for i in range(10):
              ...
          return _
      """;
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  @Test
  void loopQuickFixIndexFileLevelAlreadyTakenTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      _ = 42
      def foo():
        for i in range(5):
          print("hello")
        print(_)
      foo()
      """;
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  @Test
  void loopIndexComprehensionClassQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      class A():
        _ = True
        def __init__(self):
          for i in range(5):
            print("print") 
      """;
    var after = """
      class A():
        _ = True
        def __init__(self):
          for _ in range(5):
            print("print") 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void assignmentQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def foo():
        x = bar()
        y = True
        return y 
      """;
    var after = """
      def foo():
        bar()
        y = True
        return y 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove assignment target");
  }

  @Test
  void assignmentExpressionQuickFixTest() {
    var check = new UnusedLocalVariableCheck();
    var before = """
      def foo():
        if any((i := j) % 2 == 1 for j in range(3)):
          return 
      """;
    var after = """
      def foo():
        if any(j % 2 == 1 for j in range(3)):
          return 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove assignment target");

    PythonQuickFixVerifier.verify(check,
      """
        def foo():
          if any((i := j*2) % 2 == 1 for j in range(3)):
            return
        """,
      """
        def foo():
          if any((j*2) % 2 == 1 for j in range(3)):
            return
        """
    );

    PythonQuickFixVerifier.verify(check,
      """
        def foo():
          if (test := some_method()):
            return
        """,
      """
        def foo():
          if (some_method()):
            return
        """
    );

    PythonQuickFixVerifier.verify(check,
      """
        def foo():
          if test := some_method():
            return
        """,
      """
        def foo():
          if some_method():
            return
        """
    );
  }

  @Test
  void multipleAssignmentQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def foo():
        x, y, z = bar(), True, False
        return y, z 
      """;
    var after = """
      def foo():
        _, y, z = bar(), True, False
        return y, z 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void typeAnnotationQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def foo():
        value: str = "hello"
        return [int(value) for value in something()] 
      """;
    var after = """
      def foo():
        "hello"
        return [int(value) for value in something()] 
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove assignment target");
  }

  @Test
  void typeAnnotationSeparateDeclarationAssignmentNoQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = """
      def foo():
        value: str\s
        value = "Hello"
        return [int(value) for value in something()] 
      """;

    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }
}
