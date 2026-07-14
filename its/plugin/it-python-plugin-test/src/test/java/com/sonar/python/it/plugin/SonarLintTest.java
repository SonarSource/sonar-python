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
package com.sonar.python.it.plugin;

import java.nio.file.Path;
import java.util.Set;
import org.junit.jupiter.api.io.TempDir;
import org.sonarsource.sonarlint.core.rpc.protocol.common.Language;
import org.sonarsource.sonarlint.core.test.utils.junit5.SonarLintTestHarness;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

public class SonarLintTest {

  @org.sonarsource.sonarlint.core.test.utils.junit5.SonarLintTest
  void should_raise_issues(SonarLintTestHarness harness, @TempDir Path baseDir) {
    var inputFile = SonarLintBackendTestUtils.writeFile(baseDir, "foo.py",
      """
      def fooBar():
        `1`
        `1` #NOSONAR
      """,
      Language.PYTHON);

    var client = SonarLintBackendTestUtils.createClient(harness, baseDir, inputFile);
    var backend = SonarLintBackendTestUtils.startBackend(
      harness,
      client,
      Set.of(Language.PYTHON),
      Set.of("python:S1542", "python:BackticksUsage"));

    SonarLintBackendTestUtils.openFile(backend, inputFile);

    SonarLintBackendTestUtils.awaitIssues(client, issuesByUri -> assertThat(
      issuesByUri.entrySet().stream()
        .flatMap(entry -> entry.getValue().stream()
          .map(issue -> tuple(issue.getRuleKey(), issue.getTextRange().getStartLine(), entry.getKey())))
        .toList())
      .containsExactlyInAnyOrder(
        tuple("python:BackticksUsage", 2, inputFile.getUri()),
        tuple("python:S1542", 1, inputFile.getUri())));
  }
}
