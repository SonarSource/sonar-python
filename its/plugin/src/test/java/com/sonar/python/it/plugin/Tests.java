/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2018 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.locator.FileLocation;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.junit.ClassRule;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.sonarqube.ws.WsMeasures;
import org.sonarqube.ws.WsMeasures.Measure;
import org.sonarqube.ws.client.HttpConnector;
import org.sonarqube.ws.client.WsClient;
import org.sonarqube.ws.client.WsClientFactories;
import org.sonarqube.ws.client.measure.ComponentWsRequest;

import static com.sonar.orchestrator.container.Server.ADMIN_LOGIN;
import static com.sonar.orchestrator.container.Server.ADMIN_PASSWORD;
import static java.lang.Double.parseDouble;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

@RunWith(Suite.class)
@Suite.SuiteClasses({
  MetricsTest.class,
  CoverageTest.class,
  PylintReportTest.class,
  TestReportTest.class,
  NoSonarTest.class,
  XPathRuleTest.class,
  SonarLintTest.class
})
public class Tests {

  public static final FileLocation PLUGIN_LOCATION = FileLocation.byWildcardMavenFilename(new File("../../sonar-python-plugin/target"), "sonar-python-plugin-*.jar");

  @ClassRule
  public static Orchestrator ORCHESTRATOR = Orchestrator.builderEnv()
    .addPlugin(PLUGIN_LOCATION)
    .restoreProfileAtStartup(FileLocation.of("profiles/no_rule.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/pylint.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/nosonar.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/xpath_rule.xml"))
    .build();

  public static Integer getProjectMeasure(String projectKey, String metricKey) {
    return getMeasureAsInt(projectKey, metricKey);
  }

  public static void assertProjectMeasures(String projectKey, Map<String, Integer> expected) {
    Map<String, Measure> measuresByMetricKey = getMeasures(projectKey, new ArrayList<>(expected.keySet()))
      .stream()
      .collect(Collectors.toMap(Measure::getMetric, Function.identity()));
    for (Entry<String, Integer> entry : expected.entrySet()) {
      String metric = entry.getKey();
      Integer expectedValue = entry.getValue();
      Measure measure = measuresByMetricKey.get(metric);
      Integer value = measure == null ? null : ((Double) parseDouble(measure.getValue())).intValue();
      assertThat(value).as(metric).isEqualTo(expectedValue);
    }
  }

  @CheckForNull
  static Measure getMeasure(String componentKey, String metricKey) {
    List<Measure> measures = getMeasures(componentKey, singletonList(metricKey));
    return measures.size() == 1 ? measures.get(0) : null;
  }

  @CheckForNull
  private static List<Measure> getMeasures(String componentKey, List<String> metricKeys) {
    WsMeasures.ComponentWsResponse response = newWsClient().measures().component(new ComponentWsRequest()
      .setComponentKey(componentKey)
      .setMetricKeys(metricKeys));
    return response.getComponent().getMeasuresList();
  }

  @CheckForNull
  static Integer getMeasureAsInt(String componentKey, String metricKey) {
    Measure measure = getMeasure(componentKey, metricKey);
    return (measure == null) ? null : Integer.parseInt(measure.getValue());
  }

  @CheckForNull
  static Double getMeasureAsDouble(String componentKey, String metricKey) {
    Measure measure = getMeasure(componentKey, metricKey);
    return (measure == null) ? null : parseDouble(measure.getValue());
  }

  protected static WsClient newWsClient() {
    return newWsClient(null, null);
  }

  protected static WsClient newAdminWsClient() {
    return newWsClient(ADMIN_LOGIN, ADMIN_PASSWORD);
  }

  protected static WsClient newWsClient(@Nullable String login, @Nullable String password) {
    return WsClientFactories.getDefault().newClient(HttpConnector.newBuilder()
      .url(ORCHESTRATOR.getServer().getUrl())
      .credentials(login, password)
      .build());
  }

}
