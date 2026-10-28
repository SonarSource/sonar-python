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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class NonContextManagerCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/nonContextManager.py", new NonContextManagerCheck());
  }

  @Test
  void crossModule() {
    PythonCheckVerifier.verify(
      List.of("src/test/resources/checks/nonContextManagerImporter.py", "src/test/resources/checks/nonContextManagerImported.py"),
      new NonContextManagerCheck());
  }

  @Test
  void projectLocalContextlibDecorator() {
    PythonCheckVerifier.verifyNoIssue(
      List.of("src/test/resources/checks/nonContextManagerShadowedContextlib/consumer.py",
        "src/test/resources/checks/nonContextManagerShadowedContextlib/contextlib.py"),
      new NonContextManagerCheck());
  }

  @Test
  void asyncFactoryInsideAsyncFunctionIsReportedOnce() {
    String path = "src/test/resources/checks/nonContextManagerAsyncFactory.py";
    PythonCheckVerifier.verify(path, new NonContextManagerCheck());
    assertThat(PythonCheckVerifier.issues(path, new AsyncWithContextManagerCheck())).isEmpty();
  }

  @Test
  void backportedContextManager() {
    PythonCheckVerifier.verifyNoIssue(
      List.of("src/test/resources/checks/nonContextManagerBackportImporter.py", "src/test/resources/checks/nonContextManagerBackport.py"),
      new NonContextManagerCheck());
  }
  @Test
  void aliasedContextManagerDecorators() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/nonContextManagerDecoratorAliases.py", new NonContextManagerCheck());
  }

}
