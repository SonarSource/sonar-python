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
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.IPythonLocation;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.SubscriptionVisitor;

import static org.assertj.core.api.Assertions.assertThat;

class AsyncFunctionNotAsyncCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/asyncFunctionNotAsync.py", new AsyncFunctionNotAsyncCheck());
  }

  @Test
  void httpx_property_assignment() {
    PythonCheckVerifier.verify("src/test/resources/checks/asyncFunctionNotAsyncHttpxProperty.py", new AsyncFunctionNotAsyncCheck());
  }

  @Test
  void httpx_property_assignment_in_compressed_notebook() {
    String content = """
      import httpx
      async def hook(response):
          print(response.status_code)
      def configure():
          client = httpx.AsyncClient()
          previous_hooks = client.event_hooks
          client.event_hooks = {"response": [hook]}
      """;
    Map<Integer, IPythonLocation> locations = Map.of(
      1, new IPythonLocation(1, 1),
      2, new IPythonLocation(1, 30),
      3, new IPythonLocation(1, 75),
      4, new IPythonLocation(1, 115),
      5, new IPythonLocation(1, 145),
      6, new IPythonLocation(1, 190),
      7, new IPythonLocation(1, 240),
      8, new IPythonLocation(1, 290));
    PythonVisitorContext context = TestPythonVisitorRunner.scanNotebookFile(
      new File("src/test/resources/checks/asyncFunctionNotAsyncHttpxProperty.ipynb"), locations, content);
    SubscriptionVisitor.analyze(Collections.singletonList(new AsyncFunctionNotAsyncCheck()), context);

    assertThat(context.getIssues()).isEmpty();
  }

}
