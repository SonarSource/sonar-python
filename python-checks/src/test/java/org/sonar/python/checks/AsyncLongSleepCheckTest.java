/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class AsyncLongSleepCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/asyncLongSleep.py", new AsyncLongSleepCheck());
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFixTest(String testName, String before, String after, String expectedMessage) {
    var check = new AsyncLongSleepCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      Arguments.of("trio literal", """
        import trio

        async def f():
            await trio.sleep(86401)
        """, """
        import trio

        async def f():
            await trio.sleep_forever()
        """, "Replace with trio.sleep_forever()"),
      Arguments.of("anyio literal", """
        import anyio

        async def f():
            await anyio.sleep(86401)
        """, """
        import anyio

        async def f():
            await anyio.sleep_forever()
        """, "Replace with anyio.sleep_forever()"),
      Arguments.of("trio alias", """
        import trio as t

        async def f():
            await t.sleep(86401)
        """, """
        import trio as t

        async def f():
            await t.sleep_forever()
        """, "Replace with t.sleep_forever()"),
      Arguments.of("anyio alias", """
        import anyio as a

        async def f():
            await a.sleep(86401)
        """, """
        import anyio as a

        async def f():
            await a.sleep_forever()
        """, "Replace with a.sleep_forever()"));
  }

  @ParameterizedTest
  @MethodSource("noQuickFixTestCases")
  void noQuickFixTest(String testName, String before) {
    var check = new AsyncLongSleepCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  static Stream<Arguments> noQuickFixTestCases() {
    return Stream.of(
      Arguments.of("submodule import", """
        import trio.sleep
        async def f():
            await trio.sleep(86401)
        """),
      Arguments.of("from import", """
        from anyio import sleep
        async def f():
            await sleep(86401)
        """));
  }

  @Test
  void multipleLibrariesImportedTest() {
    var check = new AsyncLongSleepCheck();
    var before = """
      import trio as t
      import anyio as a

      async def f():
          await t.sleep(86401)
      """;
    var after = """
      import trio as t
      import anyio as a

      async def f():
          await t.sleep_forever()
      """;
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before,
      "Replace with t.sleep_forever()");
  }

}
