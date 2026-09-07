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

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.python.it.ConcurrentOrchestratorExtension;
import com.sonar.python.it.TestsUtils;
import java.io.File;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Issues;

import static com.sonar.python.it.TestsUtils.issues;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;

public class NotebookPluginTest {

  private static final String PROJECT_KEY = "ipynb_json_project";
  private static final String DATABRICKS_PROJECT_KEY = "databricks_notebook_project";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.dynamicOrchestrator;

  @BeforeAll
  static void startServer() {
    analyzeProject(PROJECT_KEY);
    analyzeProject(DATABRICKS_PROJECT_KEY);
  }

  private static void analyzeProject(String projectKey) {
    ORCHESTRATOR.getServer().provisionProject(projectKey, projectKey);
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File("projects", projectKey))
      .setProjectKey(projectKey)
      .setProjectName(projectKey)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".");
    assertThat(ORCHESTRATOR.executeBuild(build).getLogs())
      .doesNotContain("Unable to parse file", "Unable to analyze file");
  }

  @Test
  void test() {
    List<Issues.Issue> issues = issues(PROJECT_KEY);
    assertThat(issues)
      .extracting(Issues.Issue::getRule)
      .containsExactlyInAnyOrder("ipython:PrintStatementUsage", "ipython:S1854", "ipython:S3457", "ipython:S5727", "ipython:S5727");
  }

  @Test
  void magic_cells_preserve_python_analysis_and_locations() {
    // S3457 is active in the default notebook profile. These ranges refer to the original JSON source.
    assertThat(issues(DATABRICKS_PROJECT_KEY))
      .extracting(
        Issues.Issue::getComponent,
        Issues.Issue::getRule,
        issue -> issue.getTextRange().getStartLine(),
        issue -> issue.getTextRange().getStartOffset(),
        issue -> issue.getTextRange().getEndLine(),
        issue -> issue.getTextRange().getEndOffset())
      .containsExactlyInAnyOrder(
        tuple(DATABRICKS_PROJECT_KEY + ":databricks_magics.ipynb", "ipython:S3457", 9, 18, 9, 27),
        tuple(DATABRICKS_PROJECT_KEY + ":databricks_magics.ipynb", "ipython:S3457", 51, 17, 51, 25),
        tuple(DATABRICKS_PROJECT_KEY + ":jupyter_line_magic.ipynb", "ipython:S3457", 10, 23, 10, 33));
  }
}
