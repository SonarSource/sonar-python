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

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
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


  private record QuickFixTestData(String scenario, String before, String after, String msg) {
    public QuickFixTestData(String scenario, String before) {
      this(scenario, before, before, "");
    }
  }

  public static Stream<QuickFixTestData> verifyQuickFixData() {
    return Stream.of(
      new QuickFixTestData(
        "tuple1",
        """
          def using_tuples():
            x, y = (1, 2)
            print x
          """,
        """
          def using_tuples():
            x, _ = (1, 2)
            print x 
          """,
        "Replace with \"_\""),
      new QuickFixTestData(
        "tuple2",
        """
          def using_tuples():
              x, y = (1, 2)
              y = 5
              print x 
          """,
        """
          def using_tuples():
              x, _ = (1, 2)
              y = 5
              print x 
          """,
        "Replace with \"_\""
      ),
      new QuickFixTestData(
        "exceptClause",
        """
          def foo():
            try:
              ...
            except Type as e:
              ...
          """,
        """
          def foo():
            try:
              ...
            except Type:
              ...
          """,
        "Remove the unused local variable"
      ),
      new QuickFixTestData(
        "loopIndex",
        """
          def loop_index():
            for i in range(10):
              print("Hello")
          """,
        """
          def loop_index():
            for _ in range(10):
              print("Hello")
          """,
        "Replace with \"_\""
      ),
      new QuickFixTestData(
        "loopIndexComprehension",
        """
          def loop_index():
           return [True for i in range(10)]
          """,
        """
          def loop_index():
           return [True for _ in range(10)]
          """,
        "Replace with \"_\""
      ),
      new QuickFixTestData(
        "loopIndexComprehensionClass",
        """
          class A():
            _ = True
            def __init__(self):
              for i in range(5):
                print("print")
          """,
        """
          class A():
            _ = True
            def __init__(self):
              for _ in range(5):
                print("print")
          """,
        "Replace with \"_\""
      ),
      new QuickFixTestData(
        "assignment",
        """
          def foo():
            x = bar()
            y = True
            return y 
          """,
        """
          def foo():
            bar()
            y = True
            return y 
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData(
        "assignmentExpression1",
        """
          def foo():
            if any((i := j) % 2 == 1 for j in range(3)):
              return 
          """,
        """
          def foo():
            if any(j % 2 == 1 for j in range(3)):
              return 
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData("assignmentExpression2",
        """
          def foo():
            if any((i := j*2) % 2 == 1 for j in range(3)):
              return
          """,
        """
          def foo():
            if any((j*2) % 2 == 1 for j in range(3)):
              return
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData("assignmentExpression3",
        """
          def foo():
            if (test := some_method()):
              return
          """,
        """
          def foo():
            if (some_method()):
              return
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData("assignmentExpression4",
        """
          def foo():
            if test := some_method():
              return
          """,
        """
          def foo():
            if some_method():
              return
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData("setAssignmentExpression",
        """
          def foo():
            if {(i := j*2) for j in range(3)}:
              return
          """,
        """
          def foo():
            if {(j*2) for j in range(3)}:
              return
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData("dictAssignmentExpression1",
        """
          def foo():
            if {'test':(i := j*2) for j in range(3)}:
              return
          """,
        """
          def foo():
            if {'test':(j*2) for j in range(3)}:
              return
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData("dictAssignmentExpression2",
        """
          def foo():
            if {(i := j*2):'test' for j in range(3)}:
              return
          """,
        """
          def foo():
            if {(j*2):'test' for j in range(3)}:
              return
          """,
        "Remove assignment target"
      ),
      new QuickFixTestData(
        "multipleAssignment",
        """
          def foo():
            x, y, z = bar(), True, False
            return y, z 
          """,
        """
          def foo():
            _, y, z = bar(), True, False
            return y, z 
          """,
        "Replace with \"_\""
      ),
      new QuickFixTestData(
        "typeAnnotation",
        """
          def foo():
            value: str = "hello"
            return [int(value) for value in something()] 
          """,
        """
          def foo():
            "hello"
            return [int(value) for value in something()] 
          """,
        "Remove assignment target")
    );
  }

  @ParameterizedTest
  @MethodSource("verifyQuickFixData")
  void verifyQuickFix(QuickFixTestData testData) {
    var check = new UnusedLocalVariableCheck();
    PythonQuickFixVerifier.verify(check, testData.before(), testData.after());
    PythonQuickFixVerifier.verifyQuickFixMessages(check, testData.before(), testData.msg());
  }

  static Stream<QuickFixTestData> verifyNoQuickFixData() {
    return Stream.of(
      new QuickFixTestData("loopQuickFixIndexAlreadyTaken",
        """
          def a():
              _ = 3
              for i in range(10):
                  ...
              return _
          """
      ),
      new QuickFixTestData("loopQuickFixIndexFileLevelAlreadyTaken",
        """
          _ = 42
          def foo():
            for i in range(5):
              print("hello")
            print(_)
          foo()
          """
      ),
      new QuickFixTestData(
        "typeAnnotationSeparateDeclarationAssignmentNoQuickFix",
        """
          def foo():
            value: str\s
            value = "Hello"
            return [int(value) for value in something()] 
          """
      )
    );
  }

  @ParameterizedTest
  @MethodSource("verifyNoQuickFixData")
  void verifyNoQuickFix(QuickFixTestData testData) {
    var check = new UnusedLocalVariableCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, testData.before());
  }
}
