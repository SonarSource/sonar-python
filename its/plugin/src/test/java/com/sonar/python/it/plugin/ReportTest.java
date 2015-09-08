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
import com.sonar.orchestrator.build.SonarRunner;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonar.wsclient.Sonar;
import org.sonar.wsclient.services.Measure;
import org.sonar.wsclient.services.Resource;
import org.sonar.wsclient.services.ResourceQuery;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;

public class ReportTest {

  @ClassRule
  public static Orchestrator ORCHESTRATOR = Tests.ORCHESTRATOR;

  private static final String PROJECT_KEY = "reports";

  private static Sonar wsClient;

  @BeforeClass
  public static void runAnalysis() {
    SonarRunner build = SonarRunner.create()
      .setProjectDir(new File("projects/reports"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs("src")
      .setProperty("sonar.dynamicAnalysis", "false")
      .setProperty("sonar.python.coverage.reportPath", "**/cov*.xml");
    ORCHESTRATOR.executeBuild(build);

    wsClient = ORCHESTRATOR.getServer().getWsClient();
  }

  @Test
  public void project_level() {
    assertThat(getProjectMeasure("lines_to_cover").getIntValue()).isEqualTo(5);
    assertThat(getProjectMeasure("uncovered_lines").getIntValue()).isEqualTo(2);
  }

  private Measure getProjectMeasure(String metricKey) {
    Resource resource = wsClient.find(ResourceQuery.createForMetrics(PROJECT_KEY, metricKey));
    return resource == null ? null : resource.getMeasure(metricKey);
  }
}
