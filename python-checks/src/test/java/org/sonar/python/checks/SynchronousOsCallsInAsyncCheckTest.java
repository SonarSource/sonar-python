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

class SynchronousOsCallsInAsyncCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/synchronousOsCallsInAsync.py", new SynchronousOsCallsInAsyncCheck());
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      // Trio test
      Arguments.of(
        "Wrap with \"await trio.thread.executor\".",
        """
          import trio
          import os
          
          async def foo():
              os.waitpid(123, 0)
          """,
        """
          import trio
          import os
          
          async def foo():
              await trio.to_thread.run_sync(os.waitpid, 123, 0)
          """
      ),
      // Anyio test
      Arguments.of(
        "Wrap with \"await anyio.thread.executor\".",
        """
          import anyio
          import os
          
          async def foo():
              os.waitpid(123, 0)
          """,
        """
          import anyio
          import os
          
          async def foo():
              await anyio.to_thread.run_sync(os.waitpid, 123, 0)
          """
      ),
      // Test with keyword arguments
      Arguments.of(
        "Wrap with \"await trio.thread.executor\".",
        """
          import trio
          import os
          
          async def foo():
              os.waitpid(pid=123, options=0)
          """,
        """
          import trio
          import os
          
          async def foo():
              await trio.to_thread.run_sync(os.waitpid, pid=123, options=0)
          """
      ),
      // Test with alias
      Arguments.of(
        "Wrap with \"await trio.thread.executor\".",
        """
          import trio as t
          import os
          
          async def foo():
              os.waitpid(123, 0)
          """,
        """
          import trio as t
          import os
          
          async def foo():
              await t.to_thread.run_sync(os.waitpid, 123, 0)
          """
      )
    );
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFixTest(String expectedMessage, String before, String after) {
    var check = new SynchronousOsCallsInAsyncCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }
  
  static Stream<Arguments> noQuickFixTestCases() {
    return Stream.of(
      // Submodule import test
      Arguments.of(
        """
          import trio.lowlevel
          import os
          
          async def foo():
              os.waitpid(123, 0)  # Noncompliant
          """
      ),
      // From import test
      Arguments.of(
        """
          from trio import sleep
          import os
          
          async def foo():
              os.waitpid(123, 0)  # Noncompliant
          """
      ),
      // No import test
      Arguments.of(
        """
          import os
          
          async def foo():
              os.waitpid(123, 0)  # Noncompliant
          """
      )
    );
  }

  @ParameterizedTest
  @MethodSource("noQuickFixTestCases")
  void noQuickFixTest(String code) {
    var check = new SynchronousOsCallsInAsyncCheck();
    PythonQuickFixVerifier.verifyNoQuickFixes(check, code);
  }

  @Test
  void multipleImportsTest() {
    var check = new SynchronousOsCallsInAsyncCheck();
    var before = """
      import trio
      import anyio
      import os
      
      async def foo():
          os.waitpid(123, 0)
      """;

    var afterWithTrio = """
      import trio
      import anyio
      import os
      
      async def foo():
          await trio.to_thread.run_sync(os.waitpid, 123, 0)
      """;

    var afterWithAnyio = """
      import trio
      import anyio
      import os
      
      async def foo():
          await anyio.to_thread.run_sync(os.waitpid, 123, 0)
      """;

    PythonQuickFixVerifier.verify(check, before, afterWithTrio, afterWithAnyio);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, 
      "Wrap with \"await trio.thread.executor\".", 
      "Wrap with \"await anyio.thread.executor\".");
  }
}
