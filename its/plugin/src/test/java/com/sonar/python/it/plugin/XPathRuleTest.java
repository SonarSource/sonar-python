/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2017 SonarSource SA
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

import com.google.common.collect.ImmutableMap;
import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonar.wsclient.SonarClient;
import org.sonar.wsclient.issue.Issue;
import org.sonar.wsclient.issue.IssueQuery;

import static org.assertj.core.api.Assertions.assertThat;

public class XPathRuleTest {

  private static final String PROJECT = "XPathRuleTest";

  private static final String PROFILE_NAME = "xpath_rule";

  private static final String RULE_KEY = "XPathTestRuleKey";

  private static final String RULE_KEY_WITH_PREFIX = "python:" + RULE_KEY;

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  @Before
  public void resetData() throws Exception {
    orchestrator.resetData();
  }

  @Test
  public void testXPathRule() {
    SonarClient sonarClient = orchestrator.getServer().adminWsClient();
    sonarClient.post("/api/rules/create", ImmutableMap.<String, Object>builder()
      .put("name", "XPathTestRule")
      .put("markdown_description", "XPath test rule")
      .put("severity", "INFO")
      .put("status", "READY")
      .put("template_key", "python:XPath")
      .put("custom_key", RULE_KEY)
      .put("prevent_reactivation", "true")
      .put("params", "message=\"Do something fantastic!\";xpathQuery=\"//FILE_INPUT\"")
      .build());
    String profiles = sonarClient.get("api/rules/app");
    Pattern pattern = Pattern.compile("py-" + PROFILE_NAME + "-\\d+");
    Matcher matcher = pattern.matcher(profiles);
    assertThat(matcher.find()).isTrue();
    String profilekey = matcher.group();
    sonarClient.post("api/qualityprofiles/activate_rule", ImmutableMap.<String, Object>of(
      "profile_key", profilekey,
      "rule_key", RULE_KEY_WITH_PREFIX,
      "severity", "INFO",
      "params", ""));

    orchestrator.getServer().provisionProject(PROJECT, PROJECT);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT, "py", PROFILE_NAME);
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/xpath_rule_project"))
      .setProjectKey(PROJECT)
      .setProjectName(PROJECT)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".");
    orchestrator.executeBuild(build);

    List<Issue> issues = getIssues(RULE_KEY_WITH_PREFIX);
    assertThat(issues.size()).isEqualTo(1);
    Issue issue = issues.get(0);
    assertThat(issue.componentKey()).endsWith("myClass.py");
    assertThat(issue.line()).isEqualTo(1);
    assertThat(issue.message()).isEqualTo("Do something fantastic!");
  }

  private List<Issue> getIssues(String ruleKey) {
    IssueQuery query = IssueQuery.create().componentRoots(PROJECT).rules(ruleKey);
    return orchestrator.getServer().wsClient().issueClient().find(query).list();
  }

}
