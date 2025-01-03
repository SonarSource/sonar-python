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

import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.Measures.ComponentWsResponse;
import org.sonarqube.ws.Measures.Measure;
import org.sonarqube.ws.client.HttpConnector;
import org.sonarqube.ws.client.WsClient;
import org.sonarqube.ws.client.WsClientFactories;
import org.sonarqube.ws.client.issues.SearchRequest;
import org.sonarqube.ws.client.measures.ComponentRequest;

import static java.lang.Double.parseDouble;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;


public final class TestsUtils {

  private static final String SQ_VERSION_PROPERTY = "sonar.runtimeVersion";
  private static final String DEFAULT_SQ_VERSION = "LATEST_RELEASE";

  public static final FileLocation PLUGIN_LOCATION = FileLocation.byWildcardMavenFilename(new File("../../../sonar-python-plugin/target"), "sonar-python-plugin-*.jar");

  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = ConcurrentOrchestratorExtension.builderEnv()
    .useDefaultAdminCredentialsForBuilds(true)
    .setSonarVersion(System.getProperty(SQ_VERSION_PROPERTY, DEFAULT_SQ_VERSION))
    .addPlugin(PLUGIN_LOCATION)
    .setServerProperty("sonar.telemetry.enable", "false") // Disable telemetry waiting for ORCH-497

    // Custom rules plugin
    .addPlugin(FileLocation.byWildcardMavenFilename(new File("../python-custom-rules-plugin/target"), "python-custom-rules-plugin-*.jar"))
    .addPlugin(FileLocation.byWildcardMavenFilename(new File("../../../docs/python-custom-rules-example/target"), "python-custom-rules-example-*.jar"))
    .restoreProfileAtStartup(FileLocation.of("profiles/profile-python-custom-rules-example.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/profile-python-custom-rules.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/profile-python-test-rules.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/no_rule.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/pylint.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/nosonar.xml"))
    .build();

  private TestsUtils() {
    // utility class
  }

  public static Integer getProjectMeasure(String projectKey, String metricKey) {
    return getMeasureAsInt(projectKey, metricKey);
  }

  public static void assertProjectMeasures(String projectKey, Map<String, Integer> expected) {
    List<Measure> measures = getMeasures(projectKey, new ArrayList<>(expected.keySet()));
    assertThat(measures).isNotNull();
    Map<String, Measure> measuresByMetricKey = measures.stream()
      .collect(Collectors.toMap(Measure::getMetric, Function.identity()));
    for (Entry<String, Integer> entry : expected.entrySet()) {
      String metric = entry.getKey();
      Integer expectedValue = entry.getValue();
      Measure measure = measuresByMetricKey.get(metric);
      Integer value = measure == null ? null : ((Double) parseDouble(measure.getValue())).intValue();
      assertThat(value).as(metric).isEqualTo(expectedValue);
    }
  }

  static Measure getMeasure(String componentKey, String metricKey) {
    List<Measure> measures = getMeasures(componentKey, singletonList(metricKey));
    return measures != null && measures.size() == 1 ? measures.get(0) : null;
  }

  private static List<Measure> getMeasures(String componentKey, List<String> metricKeys) {
    ComponentWsResponse response = newWsClient().measures().component(new ComponentRequest()
      .setComponent(componentKey)
      .setMetricKeys(metricKeys));
    return response.getComponent().getMeasuresList();
  }

  static Integer getMeasureAsInt(String componentKey, String metricKey) {
    Measure measure = getMeasure(componentKey, metricKey);
    return (measure == null) ? null : Integer.parseInt(measure.getValue());
  }

  static Double getMeasureAsDouble(String componentKey, String metricKey) {
    Measure measure = getMeasure(componentKey, metricKey);
    return (measure == null) ? null : parseDouble(measure.getValue());
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
