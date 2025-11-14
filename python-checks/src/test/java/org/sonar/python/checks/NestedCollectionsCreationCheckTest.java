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

class NestedCollectionsCreationCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/nestedCollectionsCreation.py", new NestedCollectionsCreationCheck());
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFixTest(String before, String after, String expectedMessage) {
    var check = new NestedCollectionsCreationCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @ParameterizedTest
  @MethodSource("noQuickFixTestCases")
  void noQuickFixTest(String before) {
    PythonQuickFixVerifier.verifyNoQuickFixes(new NestedCollectionsCreationCheck(), before);
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      Arguments.of(
        """
          iterable = (3, 1, 4, 1)
          list(list(iterable))
          """,
        """
          iterable = (3, 1, 4, 1)
          list(iterable)
          """,
        "Remove this redundant call."),
      Arguments.of(
        """
          iterable = (3, 1, 4, 1)
          set(list(iterable))
          """,
        """
          iterable = (3, 1, 4, 1)
          set(iterable)
          """,
        "Remove this redundant call."),
      Arguments.of(
        """
          set(list((3, 1, 4, 1)))
          """,
        """
          set((3, 1, 4, 1))
          """,
        "Remove this redundant call."),
      Arguments.of(
        """
          iterable = (3, 1, 4, 1)
          l = list(iterable)
          set(l)
          """,
        """
          iterable = (3, 1, 4, 1)
          set(iterable)
          """,
        "Remove this redundant call.")
    );
  }

  static Stream<Arguments> noQuickFixTestCases() {
    return Stream.of(
      Arguments.of(
        """
          set(list((
              3, 1,
              4, 1)))
          """)
    );
  }

}
