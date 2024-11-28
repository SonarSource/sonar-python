/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package org.sonar.python.it;

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.client.HttpConnector;
import org.sonarqube.ws.client.WsClient;
import org.sonarqube.ws.client.WsClientFactories;
import org.sonarqube.ws.client.issues.SearchRequest;
import org.sonarsource.analyzer.commons.ProfileGenerator;

import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.it.RulingHelper.getOrchestrator;

class PythonRulingTest {

  public static final String PROJECT_KEY = "project";

  @RegisterExtension
  public static final OrchestratorExtension ORCHESTRATOR = getOrchestrator();

  @BeforeAll
  static void prepare_quality_profile() {
    ProfileGenerator.RulesConfiguration parameters = new ProfileGenerator.RulesConfiguration()
      .add("CommentRegularExpression", "message", "The regular expression matches this comment")
      .add("S1451", "headerFormat", "# Copyright 2004 by Harry Zuzan. All rights reserved.");
    String serverUrl = ORCHESTRATOR.getServer().getUrl();
    File profileFile = ProfileGenerator.generateProfile(serverUrl, "py", "python", parameters, Collections.emptySet());
    ORCHESTRATOR.getServer().restoreProfile(FileLocation.of(profileFile));
    File iPythonProfileFile = ProfileGenerator.generateProfile(serverUrl, "ipynb", "ipython", parameters, Collections.emptySet());
    ORCHESTRATOR.getServer().restoreProfile(FileLocation.of(iPythonProfileFile));
  }

  @Test
  void test() throws Exception {
    ORCHESTRATOR.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "rules");
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT_KEY, "ipynb", "rules");
    File litsDifferencesFile = FileLocation.of("target/differences").getFile();
    SonarScanner build = SonarScanner.create(FileLocation.of("../sources").getFile())
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1")
      .setSourceEncoding("UTF-8")
      .setSourceDirs(".")
      .setProperty("sonar.lits.dump.old", FileLocation.of("src/test/resources/expected").getFile().getAbsolutePath())
      .setProperty("sonar.lits.dump.new", FileLocation.of("target/actual").getFile().getAbsolutePath())
      .setProperty("sonar.cpd.exclusions", "**/*")
      .setProperty("sonar.lits.differences", litsDifferencesFile.getAbsolutePath())
      .setProperty("sonar.internal.analysis.failFast", "true")
      .setEnvironmentVariable("SONAR_RUNNER_OPTS", "-Xmx2000m");
    ORCHESTRATOR.executeBuild(build);

    String issueDifferences = issues(PROJECT_KEY).stream()
      .map(i -> String.join("\t", i.getRule(), "" + i.getSeverity(), i.getComponent(), "" + i.getLine()))
      .collect(Collectors.joining("\n"));
    assertThat(issueDifferences).isEmpty();

    String litsDifferences = new String(Files.readAllBytes(litsDifferencesFile.toPath()), UTF_8);
    assertThat(litsDifferences).isEmpty();
  }

  static WsClient newWsClient() {
    return newWsClient(null, null);
  }

  static WsClient newWsClient(String login, String password) {
    return WsClientFactories.getDefault().newClient(HttpConnector.newBuilder()
      .url(ORCHESTRATOR.getServer().getUrl())
      .credentials(login, password)
      .build());
  }

  static List<Issues.Issue> issues(String projectKey) {
    return newWsClient().issues().search(new SearchRequest().setProjects(singletonList(projectKey))).getIssuesList();
  }

}
