/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package com.sonar.python.it;

import com.sonar.orchestrator.container.Edition;
import com.sonar.orchestrator.locator.FileLocation;
import com.sonar.python.it.PluginLocator.Plugins;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.Measures;
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

  public static final ConcurrentOrchestratorExtension dynamicOrchestrator = makeDynamicOrchestrator();

  private static ConcurrentOrchestratorExtension makeDynamicOrchestrator() {
    if (PluginLocator.isEnterpriseTest()) {
      return orchestratorBuilder(Edition.ENTERPRISE_LW).build();
    } else {
      return orchestratorBuilder(Edition.COMMUNITY).build();
    }
  }

  private static ConcurrentOrchestratorExtension.ConcurrentOrchestratorExtensionBuilder orchestratorBuilder(Edition edition) {
    boolean isEnterprise = edition != Edition.COMMUNITY;

    var builder = ConcurrentOrchestratorExtension.builderEnv()
      .useDefaultAdminCredentialsForBuilds(true)
      .setSonarVersion(System.getProperty(SQ_VERSION_PROPERTY, DEFAULT_SQ_VERSION))
      // Disable telemetry waiting for ORCH-497
      .setServerProperty("sonar.telemetry.enable", "false")

      // Custom rules plugin
      .addPlugin(Plugins.PYTHON_CUSTOM_RULES.get(isEnterprise))
      .addPlugin(Plugins.PYTHON_CUSTOM_RULES_EXAMPLE.get(isEnterprise))
      .restoreProfileAtStartup(FileLocation.of("profiles/profile-python-custom-rules-example.xml"))
      .restoreProfileAtStartup(FileLocation.of("profiles/profile-python-custom-rules.xml"))
      .restoreProfileAtStartup(FileLocation.of("profiles/profile-python-test-rules.xml"))
      .restoreProfileAtStartup(FileLocation.of("profiles/no_rule.xml"))
      .restoreProfileAtStartup(FileLocation.of("profiles/pylint.xml"))
      .restoreProfileAtStartup(FileLocation.of("profiles/nosonar.xml"));

    if (isEnterprise) {
      builder.setEdition(edition)
        .activateLicense();
    }

    builder.addPlugin(Plugins.PYTHON.get(isEnterprise));

    return builder;
  }


  private TestsUtils() {
    // utility class
  }

  public static Integer getProjectMeasure(String projectKey, String metricKey) {
    return getMeasureAsInt(projectKey, metricKey);
  }

  public static void assertProjectMeasures(String projectKey, Map<String, Integer> expected) {
    List<Measures.Measure> measures = getMeasures(projectKey, new ArrayList<>(expected.keySet()));
    assertThat(measures).isNotNull();
    Map<String, Measures.Measure> measuresByMetricKey = measures.stream()
      .collect(Collectors.toMap(Measures.Measure::getMetric, Function.identity()));
    for (Map.Entry<String, Integer> entry : expected.entrySet()) {
      String metric = entry.getKey();
      Integer expectedValue = entry.getValue();
      Measures.Measure measure = measuresByMetricKey.get(metric);
      Integer value = measure == null ? null : ((Double) parseDouble(measure.getValue())).intValue();
      assertThat(value).as(metric).isEqualTo(expectedValue);
    }
  }

  static Measures.Measure getMeasure(String componentKey, String metricKey) {
    List<Measures.Measure> measures = getMeasures(componentKey, singletonList(metricKey));
    return measures != null && measures.size() == 1 ? measures.get(0) : null;
  }

  private static List<Measures.Measure> getMeasures(String componentKey, List<String> metricKeys) {
    Measures.ComponentWsResponse response = newWsClient().measures().component(new ComponentRequest()
      .setComponent(componentKey)
      .setMetricKeys(metricKeys));
    return response.getComponent().getMeasuresList();
  }

  public static Integer getMeasureAsInt(String componentKey, String metricKey) {
    Measures.Measure measure = getMeasure(componentKey, metricKey);
    return (measure == null) ? null : Integer.parseInt(measure.getValue());
  }

  public static Double getMeasureAsDouble(String componentKey, String metricKey) {
    Measures.Measure measure = getMeasure(componentKey, metricKey);
    return (measure == null) ? null : parseDouble(measure.getValue());
  }

  public static WsClient newWsClient() {
    return newWsClient(null, null);
  }

  static WsClient newWsClient(String login, String password) {
    return WsClientFactories.getDefault().newClient(HttpConnector.newBuilder()
      .url(dynamicOrchestrator.getServer().getUrl())
      .credentials(login, password)
      .build());
  }

  public static List<Issues.Issue> issues(String projectKey) {
    return newWsClient().issues().search(new SearchRequest().setProjects(singletonList(projectKey))).getIssuesList();
  }
}
