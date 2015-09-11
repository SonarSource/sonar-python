/*
 * Python :: ITs :: Plugin
 * Copyright (C) 2012 SonarSource and Waleri Enns
 * sonarqube@googlegroups.com
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.locator.FileLocation;
import org.junit.ClassRule;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.sonar.wsclient.services.Resource;
import org.sonar.wsclient.services.ResourceQuery;

import java.util.Map;
import java.util.Map.Entry;

import static org.fest.assertions.Assertions.assertThat;

@RunWith(Suite.class)
@Suite.SuiteClasses({
  MetricsTest.class,
  CoverageTest.class,
  PylintReportTest.class,
  TestReportTest.class
})
public class Tests {

  @ClassRule
  public static Orchestrator ORCHESTRATOR = Orchestrator.builderEnv()
    .addPlugin(FileLocation.of("../../sonar-python-plugin/target/sonar-python-plugin.jar"))
    .restoreProfileAtStartup(FileLocation.of("profiles/no_rule.xml"))
    .restoreProfileAtStartup(FileLocation.of("profiles/pylint.xml"))
    .build();

  public static Integer getProjectMeasure(String projectKey, String metricKey) {
    Resource resource = ORCHESTRATOR.getServer().getWsClient().find(ResourceQuery.createForMetrics(projectKey, metricKey));
    return resource == null ? null : resource.getMeasure(metricKey).getIntValue();
  }

  public static void assertProjectMeasures(String projectKey, Map<String, Integer> expected) {
    for (Entry<String, Integer> entry : expected.entrySet()) {
      String metric = entry.getKey();
      assertThat(getProjectMeasure(projectKey, metric)).as(metric).isEqualTo(entry.getValue());
    }
  }

}
