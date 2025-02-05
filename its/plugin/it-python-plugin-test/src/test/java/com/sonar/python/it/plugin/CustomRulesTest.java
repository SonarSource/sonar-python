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
import com.sonar.python.it.ConcurrentOrchestratorExtension;
import com.sonar.python.it.TestsUtils;
import java.io.File;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.parallel.ResourceLock;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.client.issues.SearchRequest;

import static com.sonar.python.it.TestsUtils.newWsClient;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

@ResourceLock("project/custom_rules")
public class CustomRulesTest {

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension orchestrator = TestsUtils.dynamicOrchestrator;

  private static final String PROJECT_KEY = "custom-rules";
  private static final String PROJECT_NAME = "Custom Rules";

  @BeforeAll
  static void prepare() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_NAME);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "python-custom-rules-profile");
    SonarScanner build = orchestrator.createSonarScanner()
      .setProjectDir(new File("projects/custom_rules"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_NAME)
      .setProjectVersion("1.0")
      .setSourceDirs("src");
    orchestrator.executeBuild(build);
  }

  @Test
  void base_tree_visitor_check() {
    List<Issues.Issue> issues = issues("python-custom-rules:visitor");
    assertSingleIssue(issues, 4, "Function def.", "5min");
  }

  @Test
  void subscription_base_visitor_check() {
    List<Issues.Issue> issues = issues("python-custom-rules:subscription");
    assertSingleIssue(issues, 7, "For statement.", "10min");
  }

  private void assertSingleIssue(List<Issues.Issue> issues, int expectedLine, String expectedMessage, String expectedDebt) {
    assertThat(issues).hasSize(1);
    Issues.Issue issue = issues.get(0);
    assertThat(issue.getLine()).isEqualTo(expectedLine);
    assertThat(issue.getMessage()).isEqualTo(expectedMessage);
    assertThat(issue.getDebt()).isEqualTo(expectedDebt);
  }

  private static List<Issues.Issue> issues(String rulekey) {
    return newWsClient().issues().search(new SearchRequest().setRules(singletonList(rulekey))).getIssuesList();
  }
}
