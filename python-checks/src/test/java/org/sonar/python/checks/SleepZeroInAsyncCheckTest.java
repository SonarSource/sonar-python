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

class SleepZeroInAsyncCheckTest {
  @Test
  void testTrio() {
    PythonCheckVerifier.verify("src/test/resources/checks/sleepZeroInAsyncTrio.py", new SleepZeroInAsyncCheck());
  }

  @Test
  void testAnyio() {
    PythonCheckVerifier.verify("src/test/resources/checks/sleepZeroInAsyncAnyio.py", new SleepZeroInAsyncCheck());
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFixTest(String testName, String before, String after, String expectedMessage) {
    var check = new SleepZeroInAsyncCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @ParameterizedTest
  @MethodSource("noQuickFixTestCases")
  void noQuickFixTest(String testName, String before) {
    var check = new SleepZeroInAsyncCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      Arguments.of(
        "trio sleep with integer zero",
        """
          import trio
          async def test():
              await trio.sleep(0)""",
        """
          import trio
          async def test():
              await trio.lowlevel.checkpoint()""",
        "Replace with trio.lowlevel.checkpoint()"),
      Arguments.of(
        "trio sleep with float zero",
        """
          import trio
          async def test():
              await trio.sleep(0.0)""",
        """
          import trio
          async def test():
              await trio.lowlevel.checkpoint()""",
        "Replace with trio.lowlevel.checkpoint()"),
      Arguments.of(
        "trio sleep with keyword argument",
        """
          import trio
          async def test():
              await trio.sleep(seconds=0)""",
        """
          import trio
          async def test():
              await trio.lowlevel.checkpoint()""",
        "Replace with trio.lowlevel.checkpoint()"),
      Arguments.of(
        "trio sleep with alias",
        """
          import trio as t
          async def test():
              await t.sleep(0)""",
        """
          import trio as t
          async def test():
              await t.lowlevel.checkpoint()""",
        "Replace with t.lowlevel.checkpoint()"),
      Arguments.of(
        "anyio sleep with integer zero",
        """
          import anyio
          async def test():
              await anyio.sleep(0)""",
        """
          import anyio
          async def test():
              await anyio.lowlevel.checkpoint()""",
        "Replace with anyio.lowlevel.checkpoint()"),
      Arguments.of(
        "anyio sleep with float zero",
        """
          import anyio
          async def test():
              await anyio.sleep(0.0)""",
        """
          import anyio
          async def test():
              await anyio.lowlevel.checkpoint()""",
        "Replace with anyio.lowlevel.checkpoint()"),
      Arguments.of(
        "anyio sleep with keyword argument",
        """
          import anyio
          async def test():
              await anyio.sleep(delay=0)""",
        """
          import anyio
          async def test():
              await anyio.lowlevel.checkpoint()""",
        "Replace with anyio.lowlevel.checkpoint()"),
      Arguments.of(
        "anyio sleep with alias",
        """
          import anyio as a
          async def test():
              await a.sleep(0)""",
        """
          import anyio as a
          async def test():
              await a.lowlevel.checkpoint()""",
        "Replace with a.lowlevel.checkpoint()"),
      Arguments.of(
        "multiple libraries imported - trio call",
          """
          import trio
          import anyio
          async def test():
              await trio.sleep(0)""",
          """
          import trio
          import anyio
          async def test():
              await trio.lowlevel.checkpoint()""",
          "Replace with trio.lowlevel.checkpoint()"),
      Arguments.of(
        "multiple libraries imported - anyio call",
          """
          import trio
          import anyio
          async def test():
              await anyio.sleep(0)""",
          """
          import trio
          import anyio
          async def test():
              await anyio.lowlevel.checkpoint()""",
        "Replace with anyio.lowlevel.checkpoint()"));
  }

  static Stream<Arguments> noQuickFixTestCases() {
    return Stream.of(
      Arguments.of(
        "trio sleep imported directly",
        """
          from trio import sleep
          async def test():
              await sleep(0)"""),
      Arguments.of(
        "anyio sleep imported directly",
        """
          from anyio import sleep
          async def test():
              await sleep(0)"""));
  }
}
