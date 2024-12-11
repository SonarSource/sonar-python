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

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.container.Edition;
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import com.sonar.orchestrator.locator.FileLocation;
import com.sonar.orchestrator.locator.MavenLocation;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import org.sonarqube.ws.Measures;
import org.sonarqube.ws.client.HttpConnector;
import org.sonarqube.ws.client.WsClient;
import org.sonarqube.ws.client.WsClientFactories;
import org.sonarqube.ws.client.measures.ComponentRequest;

import static java.util.Collections.singletonList;

class RulingHelper {

  private static final String SQ_VERSION_PROPERTY = "sonar.runtimeVersion";
  private static final String DEFAULT_SQ_VERSION = "LATEST_RELEASE";

  static OrchestratorExtension getOrchestrator(Edition sonarEdition) {
    var builder = OrchestratorExtension.builderEnv()
      .useDefaultAdminCredentialsForBuilds(true)
      .setSonarVersion(System.getProperty(SQ_VERSION_PROPERTY, DEFAULT_SQ_VERSION))
      .setEdition(sonarEdition)
      .addPlugin(FileLocation.byWildcardMavenFilename(new File("../../sonar-python-plugin/target"), "sonar-python-plugin-*.jar"))
      .addPlugin(MavenLocation.of("org.sonarsource.sonar-lits-plugin", "sonar-lits-plugin", "0.11.0.2659"))
      .setServerProperty("sonar.telemetry.enable", "false"); // Disable telemetry waiting for ORCH-497

    if (sonarEdition != Edition.COMMUNITY) {
      builder.activateLicense();
    }

    return builder.build();
  }

  static OrchestratorExtension getOrchestrator() {
    return getOrchestrator(Edition.COMMUNITY);
  }

  static String profile(String name, String language, String repositoryKey, List<String> ruleKeys) {
    StringBuilder sb = new StringBuilder()
      .append("<profile>")
      .append("<name>").append(name).append("</name>")
      .append("<language>").append(language).append("</language>")
      .append("<rules>");
    ruleKeys.forEach(ruleKey -> {
      sb.append("<rule>")
        .append("<repositoryKey>").append(repositoryKey).append("</repositoryKey>")
        .append("<key>").append(ruleKey).append("</key>")
        .append("<priority>INFO</priority>")
        .append("</rule>");
    });

    return sb
      .append("</rules>")
      .append("</profile>")
      .toString();
  }

  static void loadProfile(Orchestrator orchestrator, String profile) throws IOException {
    File file = File.createTempFile("profile", ".xml");
    Files.write(file.toPath(), profile.getBytes());
    orchestrator.getServer().restoreProfile(FileLocation.of(file));
    file.delete();
  }

  // TODO: SONARPY-984, read rules metadata instead of hardcoding this list
  static List<String> bugRuleKeys() {
    return Arrays.asList(
      "PreIncrementDecrement",
      "S935",
      "S1045",
      "S1143",
      "S1226",
      "S1656",
      "S1716",
      "S1751",
      "S1763",
      "S1764",
      "S1862",
      "S2159",
      "S2190",
      "S2201",
      "S2275",
      "S2711",
      "S2712",
      "S2734",
      "S2757",
      "S2823",
      "S2876",
      "S3403",
      "S3699",
      "S3827",
      "S3862",
      "S3923",
      "S3981",
      "S3984",
      "S4143",
      "S5549",
      "S5607",
      "S5632",
      "S5642",
      "S5644",
      "S5707",
      "S5708",
      "S5714",
      "S5719",
      "S5722",
      "S5724",
      "S5756",
      "S5796",
      "S5807",
      "S5828",
      "S5842",
      "S5850",
      "S5855",
      "S5856",
      "S5868",
      "S5996",
      "S6002",
      "S6323",
      "S6328",
      "S6468",
      "S6662",
      "S905",
      "S930");
  }

  static Measures.Measure getMeasure(Orchestrator orchestrator, String branch, String componentKey, String metricKey) {
    List<Measures.Measure> measures = getMeasures(orchestrator, branch, componentKey, singletonList(metricKey));
    return measures != null && measures.size() == 1 ? measures.get(0) : null;
  }

  private static List<Measures.Measure> getMeasures(Orchestrator orchestrator, String prKey, String componentKey, List<String> metricKeys) {
    Measures.ComponentWsResponse response = newWsClient(orchestrator).measures().component(new ComponentRequest()
      .setComponent(componentKey)
      .setPullRequest(prKey)
      .setMetricKeys(metricKeys));
    return response.getComponent().getMeasuresList();
  }

  static WsClient newWsClient(Orchestrator orchestrator) {
    return WsClientFactories.getDefault().newClient(HttpConnector.newBuilder()
      .url(orchestrator.getServer().getUrl())
      .credentials(null, null)
      .build());
  }
}
