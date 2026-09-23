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
package com.sonar.python.it;

import com.sonar.orchestrator.config.Configuration;
import com.sonar.orchestrator.container.SonarDistribution;
import java.time.Duration;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

class ConcurrentOrchestratorExtensionTest {

  @Test
  void should_start_and_prepare_orchestrator_only_once() throws InterruptedException {
    TestOrchestrator orchestrator = new TestOrchestrator();

    orchestrator.beforeAll(null);
    orchestrator.beforeAll(null);

    assertThat(orchestrator.startCount).isEqualTo(1);
    assertThat(orchestrator.prepareCount).isEqualTo(1);
  }

  @Test
  void should_fail_fast_when_orchestrator_startup_failed() {
    TestOrchestrator orchestrator = new TestOrchestrator();
    orchestrator.startFailure = new IllegalStateException("Startup failed");

    assertThatThrownBy(() -> orchestrator.beforeAll(null))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Startup failed");

    assertSubsequentRequestFailsFast(orchestrator);
  }

  @Test
  void should_fail_fast_when_orchestrator_preparation_failed() {
    TestOrchestrator orchestrator = new TestOrchestrator();
    orchestrator.prepareFailure = new IllegalStateException("Preparation failed");

    assertThatThrownBy(() -> orchestrator.beforeAll(null))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Preparation failed");

    assertSubsequentRequestFailsFast(orchestrator);
  }

  private void assertSubsequentRequestFailsFast(ConcurrentOrchestratorExtension orchestrator) {
    assertTimeoutPreemptively(Duration.ofSeconds(1), () ->
      assertThatThrownBy(() -> orchestrator.beforeAll(null))
        .isInstanceOf(IllegalStateException.class)
        .hasMessage("Previous Orchestrator startup failed"));
  }

  private static class TestOrchestrator extends ConcurrentOrchestratorExtension {
    private int startCount;
    private int prepareCount;
    private RuntimeException startFailure;
    private RuntimeException prepareFailure;

    TestOrchestrator() {
      super(Configuration.createEnv(), new SonarDistribution(), null);
    }

    @Override
    public void start() {
      startCount++;
      if (startFailure != null) {
        throw startFailure;
      }
    }

    @Override
    void prepareOrchestrator() {
      prepareCount++;
      if (prepareFailure != null) {
        throw prepareFailure;
      }
    }
  }
}
