/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2025 SonarSource SA
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
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Issues;

import static com.sonar.python.it.plugin.TestsUtils.issues;
import static org.assertj.core.api.Assertions.assertThat;

class NoSonarTest {

  private static final String PROJECT_KEY = "nosonar";
  private static final String PROFILE_NAME = "nosonar";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  @BeforeAll
  static void startServer() {
    ORCHESTRATOR.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", PROFILE_NAME);
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File("projects", PROJECT_KEY))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".");
    ORCHESTRATOR.executeBuild(build);
  }

  @Test
  void test_nosonar() {
    List<Issues.Issue> issues = issues(PROJECT_KEY);
    assertThat(issues).hasSize(2);
    assertThat(issues.get(0).getRule()).isEqualTo("python:PrintStatementUsage");
    assertThat(issues.get(0).getLine()).isEqualTo(1);
    assertThat(issues.get(1).getRule()).isEqualTo("python:NoSonar");
    assertThat(issues.get(1).getLine()).isEqualTo(2);
  }
}
