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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class TimeSleepInAsyncCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/timeSleepInAsync.py", new TimeSleepInAsyncCheck());
  }

  @Test
  void quickFixAsyncioTest() {
    var check = new TimeSleepInAsyncCheck();
    var before = """
      import asyncio
      import time

      async def foo():
          time.sleep(1)
      """;

    var after = """
      import asyncio
      import time

      async def foo():
          asyncio.sleep(1)
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with asyncio.sleep()");
  }

  @Test
  void quickFixTrioTest() {
    var check = new TimeSleepInAsyncCheck();
    var before = """
      import trio
      import time

      async def foo():
          time.sleep(1)
      """;

    var after = """
      import trio
      import time

      async def foo():
          trio.sleep(1)
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with trio.sleep()");
  }

  @Test
  void quickFixAnyioTest() {
    var check = new TimeSleepInAsyncCheck();
    var before = """
      import anyio
      import time

      async def foo():
          time.sleep(1)
      """;

    var after = """
      import anyio
      import time

      async def foo():
          anyio.sleep(1)
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with anyio.sleep()");
  }

  @Test
  void quickFixWithAliasTest() {
    var check = new TimeSleepInAsyncCheck();
    var before = """
      import trio as t
      import time

      async def foo():
          time.sleep(1)
      """;

    var after = """
      import trio as t
      import time

      async def foo():
          t.sleep(1)
      """;

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with t.sleep()");
  }

  @Test
  void multipleImportsTest() {
    var check = new TimeSleepInAsyncCheck();
    var before = """
      import trio as t
      import asyncio
      import time

      async def foo():
          time.sleep(1)
      """;

    var afterWithTrio = """
      import trio as t
      import asyncio
      import time

      async def foo():
          t.sleep(1)
      """;

    var afterWithAsyncio = """
      import trio as t
      import asyncio
      import time

      async def foo():
          asyncio.sleep(1)
      """;

    PythonQuickFixVerifier.verify(check, before, afterWithTrio, afterWithAsyncio);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with t.sleep()", "Replace with asyncio.sleep()");
  }

  @Test
  void quickFixIPythonTest() {
    var check = new TimeSleepInAsyncCheck();
    var before = """
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      import asyncio
      import time

      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      async def foo():
          time.sleep(1)
      """;

    var after = """
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      import asyncio
      import time

      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      async def foo():
          asyncio.sleep(1)
      """;

    PythonQuickFixVerifier.verifyIPython(check, before, after);
    PythonQuickFixVerifier.verifyIPythonQuickFixMessages(check, before, "Replace with asyncio.sleep()");
  }

  @Test
  void noQuickFixForSubmoduleImportTest() {
    var check = new TimeSleepInAsyncCheck();
    var code = """
      import asyncio.tasks
      import time

      async def foo():
          time.sleep(1)  # Noncompliant
      """;

    PythonQuickFixVerifier.verifyNoQuickFixes(check, code);
  }

  @Test
  void noQuickFixForFromImportTest() {
    var check = new TimeSleepInAsyncCheck();
    var code = """
      from asyncio import create_task
      import time

      async def foo():
          time.sleep(1)  # Noncompliant
      """;

    PythonQuickFixVerifier.verifyNoQuickFixes(check, code);
  }

  @Test
  void noQuickFixWhenNoImport() {
    var check = new TimeSleepInAsyncCheck();
    var code = """
      import time

      async def foo():
          time.sleep(1)  # Noncompliant
      """;

    PythonQuickFixVerifier.verifyNoQuickFixes(check, code);
  }
}
