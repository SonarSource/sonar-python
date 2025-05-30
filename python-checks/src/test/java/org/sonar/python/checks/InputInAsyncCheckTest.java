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

import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class InputInAsyncCheckTest {

  @ParameterizedTest
  @MethodSource("testCases")
  void test(String testFilePath) {
    PythonCheckVerifier.verify(testFilePath, new InputInAsyncCheck());
  }

  static Stream<Arguments> testCases() {
    return Stream.of(
      Arguments.of("src/test/resources/checks/inputInAsync.py"),
      Arguments.of("src/test/resources/checks/inputInAsyncAsyncIO.py"),
      Arguments.of("src/test/resources/checks/inputInAsyncTrio.py"),
      Arguments.of("src/test/resources/checks/inputInAsyncAnyIO.py"));
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFixTest(String testName, String before, String after, String expectedMessage) {
    var check = new InputInAsyncCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @ParameterizedTest
  @MethodSource("noQuickFixTestCases")
  void noQuickFixTest(String testName, String before) {
    var check = new InputInAsyncCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      Arguments.of(
        "asyncio input",
        """
          import asyncio

          async def foo():
              input()""",
        """
          import asyncio

          async def foo():
              await asyncio.to_thread(input)""",
        "Wrap with await asyncio.to_thread(input)"),
      Arguments.of(
        "trio input",
        """
          import trio

          async def foo():
              input()""",
        """
          import trio

          async def foo():
              await trio.to_thread.run_sync(input)""",
        "Wrap with await trio.to_thread.run_sync(input)"),
      Arguments.of(
        "anyio input",
        """
          import anyio

          async def foo():
              input()""",
        """
          import anyio

          async def foo():
              await anyio.to_thread.run_sync(input)""",
        "Wrap with await anyio.to_thread.run_sync(input)"),
      Arguments.of(
        "asyncio alias",
        """
          import asyncio as a

          async def foo():
              input()""",
        """
          import asyncio as a

          async def foo():
              await a.to_thread(input)""",
        "Wrap with await a.to_thread(input)"),
      Arguments.of(
        "trio alias",
        """
          import trio as t

          async def foo():
              input()""",
        """
          import trio as t

          async def foo():
              await t.to_thread.run_sync(input)""",
        "Wrap with await t.to_thread.run_sync(input)"),
      Arguments.of(
        "anyio alias",
        """
          import anyio as a

          async def foo():
              input()""",
        """
          import anyio as a

          async def foo():
              await a.to_thread.run_sync(input)""",
        "Wrap with await a.to_thread.run_sync(input)"),
      Arguments.of(
        "asyncio input with prompt",
        """
          import asyncio

          async def foo():
              input("Enter")""",
        """
          import asyncio

          async def foo():
              await asyncio.to_thread(input, "Enter")""",
        "Wrap with await asyncio.to_thread(input, \"Enter\")"),
      Arguments.of(
        "asyncio input with prompt keyword",
        """
          import asyncio

          async def foo():
              input(prompt="Enter")""",
        """
          import asyncio

          async def foo():
              await asyncio.to_thread(input, prompt="Enter")""",
        "Wrap with await asyncio.to_thread(input, prompt=\"Enter\")"),
      Arguments.of(
        "trio input with prompt",
        """
          import trio

          async def foo():
              input("Enter")""",
        """
          import trio

          async def foo():
              await trio.to_thread.run_sync(input, "Enter")""",
        "Wrap with await trio.to_thread.run_sync(input, \"Enter\")"),
      Arguments.of(
        "anyio input with prompt keyword",
        """
          import anyio

          async def foo():
              input(prompt="Enter")""",
        """
          import anyio

          async def foo():
              await anyio.to_thread.run_sync(input, prompt="Enter")""",
        "Wrap with await anyio.to_thread.run_sync(input, prompt=\"Enter\")"));
  }

  static Stream<Arguments> noQuickFixTestCases() {
    return Stream.of(
      Arguments.of(
        "no async import",
        """
          import time

          async def foo():
              input()"""),
      Arguments.of(
        "submodule import",
        """
          import asyncio.tasks
          import time

          async def foo():
              input()"""),
      Arguments.of(
        "from import",
        """
          from asyncio import create_task
          import time

          async def foo():
              input()"""),
      Arguments.of(
        "renamed input",
          """
          import asyncio

          async def foo():
              something = input
              something()"""));
  }

  @ParameterizedTest
  @MethodSource("multiQuickFixTestCases")
  void multipleQuickFixTest(String testName, String before, String after1, String after2, String expectedMessage1, String expectedMessage2) {
    var check = new InputInAsyncCheck();
    PythonQuickFixVerifier.verify(check, before, after1, after2);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage1, expectedMessage2);
  }

  static Stream<Arguments> multiQuickFixTestCases() {
    return Stream.of(
      Arguments.of(
        "asyncio and trio imports",
        """
          import asyncio
          import trio

          async def foo():
              input()""",
        """
          import asyncio
          import trio

          async def foo():
              await asyncio.to_thread(input)""",
        """
          import asyncio
          import trio

          async def foo():
              await trio.to_thread.run_sync(input)""",
        "Wrap with await asyncio.to_thread(input)",
        "Wrap with await trio.to_thread.run_sync(input)"),
      Arguments.of(
        "asyncio alias and anyio alias imports",
        """
          import asyncio as a
          import anyio as b

          async def foo():
              input()""",
        """
          import asyncio as a
          import anyio as b

          async def foo():
              await a.to_thread(input)""",
        """
          import asyncio as a
          import anyio as b

          async def foo():
              await b.to_thread.run_sync(input)""",
        "Wrap with await a.to_thread(input)",
        "Wrap with await b.to_thread.run_sync(input)"),
      Arguments.of(
        "asyncio and trio imports with prompt",
        """
          import asyncio
          import trio

          async def foo():
              input("Enter")""",
        """
          import asyncio
          import trio

          async def foo():
              await asyncio.to_thread(input, "Enter")""",
        """
          import asyncio
          import trio

          async def foo():
              await trio.to_thread.run_sync(input, "Enter")""",
        "Wrap with await asyncio.to_thread(input, \"Enter\")",
        "Wrap with await trio.to_thread.run_sync(input, \"Enter\")"),
      Arguments.of(
        "asyncio and trio imports with prompt keyword",
        """
          import asyncio
          import trio

          async def foo():
              input(prompt="Enter")""",
        """
          import asyncio
          import trio

          async def foo():
              await asyncio.to_thread(input, prompt="Enter")""",
        """
          import asyncio
          import trio

          async def foo():
              await trio.to_thread.run_sync(input, prompt="Enter")""",
        "Wrap with await asyncio.to_thread(input, prompt=\"Enter\")",
        "Wrap with await trio.to_thread.run_sync(input, prompt=\"Enter\")"));
  }
}
