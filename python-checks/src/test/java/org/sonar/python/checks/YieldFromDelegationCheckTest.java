/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.io.File;
import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Named;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.IPythonLocation;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.SubscriptionVisitor;

import static org.assertj.core.api.Assertions.assertThat;

class YieldFromDelegationCheckTest {

  private static final YieldFromDelegationCheck check = new YieldFromDelegationCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/yieldFromDelegation.py", check);
  }

  // In a notebook, token line()/column() are positions in the .ipynb JSON while pythonFile().content() holds the
  // generated Python, so a check reading the source must use pythonLine()/pythonColumn() to stay on the right line.
  @Test
  void notebook_does_not_flag_a_one_element_tuple_yield() {
    String content = """
      def f(items):
          for value in items:
              yield value,
      """;
    Map<Integer, IPythonLocation> locations = Map.of(
      1, new IPythonLocation(1, 200),
      2, new IPythonLocation(1, 220),
      3, new IPythonLocation(1, 248),
      4, new IPythonLocation(1, 272));
    PythonVisitorContext context = TestPythonVisitorRunner.scanNotebookFile(
      new File("src/test/resources/checks/yieldFromDelegation.ipynb"), locations, content);
    SubscriptionVisitor.analyze(Collections.singletonList(new YieldFromDelegationCheck()), context);

    assertThat(context.getIssues()).isEmpty();
  }

  @Test
  void quick_fix_message() {
    String codeWithIssue = """
      def relay(values):
          for value in values:
              yield value
      """;
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, "Replace this loop with \"yield from\"");
  }

  @ParameterizedTest
  @MethodSource("quickFixProvider")
  void quick_fix(String codeWithIssue, String fixedCode) {
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  private static Stream<Arguments> quickFixProvider() {
    return Stream.of(
      Arguments.of(
        Named.of("simple loop", """
          def relay(values):
              for value in values:
                  yield value
          """),
        """
          def relay(values):
              yield from values
          """),
      Arguments.of(
        Named.of("surrounding statements are kept", """
          def relay(values):
              yield "start"
              for value in values:
                  yield value
              yield "end"
          """),
        """
          def relay(values):
              yield "start"
              yield from values
              yield "end"
          """),
      Arguments.of(
        Named.of("loop written on a single line", """
          def relay(values):
              for value in values: yield value
          """),
        """
          def relay(values):
              yield from values
          """),
      Arguments.of(
        Named.of("call expression as iterable", """
          def relay(paths, timeout):
              for result in search(paths, timeout=timeout):
                  yield result
          """),
        """
          def relay(paths, timeout):
              yield from search(paths, timeout=timeout)
          """),
      Arguments.of(
        Named.of("comment trailing the yield is kept", """
          def relay(values):
              for value in values:
                  yield value  # relay
          """),
        """
          def relay(values):
              yield from values  # relay
          """),
      Arguments.of(
        Named.of("nested loop keeps its indentation", """
          def read_files(paths):
              for path in paths:
                  with open(path) as file:
                      for line in file:
                          yield line
          """),
        """
          def read_files(paths):
              for path in paths:
                  with open(path) as file:
                      yield from file
          """));
  }

  @ParameterizedTest
  @MethodSource("noQuickFixProvider")
  void no_quick_fix(String codeWithIssue) {
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  private static Stream<Arguments> noQuickFixProvider() {
    return Stream.of(
      Arguments.of(Named.of("multiline iterable", """
        def relay(paths, timeout):
            for result in search(paths,
                                 timeout=timeout):
                yield result
        """)),
      Arguments.of(Named.of("implicit tuple", """
        def relay(first, second):
            for value in first, second:
                yield value
        """)),
      Arguments.of(Named.of("comment above the yield would be lost", """
        def relay(values):
            for value in values:
                # explain why
                yield value
        """)),
      Arguments.of(Named.of("comment in the loop header would be lost", """
        def relay(values):
            for value in values:  # header note
                yield value
        """)));
  }
}
