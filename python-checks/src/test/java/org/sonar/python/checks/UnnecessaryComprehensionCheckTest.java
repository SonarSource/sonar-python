/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UnnecessaryComprehensionCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/unnecessaryComprehension.py", new UnnecessaryComprehensionCheck());
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFixTest(String before, String after, String expectedMessage) {
    var check = new UnnecessaryComprehensionCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @ParameterizedTest
  @MethodSource("noQuickFixTestCases")
  void noQuickFixTest(String before) {
    PythonQuickFixVerifier.verifyNoQuickFixes(new UnnecessaryComprehensionCheck(), before);
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      Arguments.of(
        """
          [x for x in some_iterable]
          """,
        """
          list(some_iterable)
          """,
        "Replace with collection constructor call"),
      Arguments.of(
        """
          {x for x in some_iterable}
          """,
        """
          set(some_iterable)
          """,
        "Replace with collection constructor call"),
      Arguments.of(
        """
          iterable_pairs = [('a', 1), ('b', 2)]
          {k: v for k, v in iterable_pairs}
          """,
        """
          iterable_pairs = [('a', 1), ('b', 2)]
          dict(iterable_pairs)
          """,
        "Replace with collection constructor call"),
      Arguments.of(
        """
          list(x for x in some_iterable)
          """,
        """
          list(some_iterable)
          """,
        "Replace with collection constructor call")
    );
  }

  static Stream<Arguments> noQuickFixTestCases() {
    return Stream.of(
      Arguments.of(
        """
          iterable_pairs = [('a', 1), ('b', 2)]
          dict_comp = {k: v for k, v in [
                  ('a', 1),
                  ('b', 2)
          ]}
          """)
    );
  }
}
