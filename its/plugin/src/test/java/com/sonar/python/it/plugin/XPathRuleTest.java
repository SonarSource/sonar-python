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
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.List;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonar.wsclient.issue.Issue;
import org.sonar.wsclient.issue.IssueQuery;
import org.sonarqube.ws.QualityProfiles;
import org.sonarqube.ws.client.PostRequest;
import org.sonarqube.ws.client.qualityprofile.SearchWsRequest;

import static com.sonar.python.it.plugin.Tests.newAdminWsClient;
import static com.sonar.python.it.plugin.Tests.newWsClient;
import static java.lang.String.format;
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
    createAndActivateRuleFromTemplate();

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

  private void createAndActivateRuleFromTemplate() {
    String language = "py";
    newAdminWsClient().wsConnector().call(new PostRequest("api/rules/create")
      .setParam("name", "XPathTestRule")
      .setParam("markdown_description", "XPath test rule")
      .setParam("severity", "INFO")
      .setParam("status", "READY")
      .setParam("template_key", "python:XPath")
      .setParam("custom_key", RULE_KEY)
      .setParam("prevent_reactivation", "true")
      .setParam("params", "message=\"Do something fantastic!\";xpathQuery=\"//FILE_INPUT\"")).failIfNotSuccessful();

    QualityProfiles.SearchWsResponse.QualityProfile qualityProfile = newWsClient().qualityProfiles().search(new SearchWsRequest()).getProfilesList().stream()
      .filter(qp -> qp.getLanguage().equals(language))
      .filter(qp -> qp.getName().equals(PROFILE_NAME))
      .findFirst().orElseThrow(() -> new IllegalStateException(format("Could not find quality profile '%s' for language '%s' ", PROFILE_NAME, language)));
    String profileKey = qualityProfile.getKey();

    newAdminWsClient().wsConnector().call(new PostRequest("api/qualityprofiles/activate_rule")
      .setParam("profile_key", profileKey)
      .setParam("rule_key", RULE_KEY_WITH_PREFIX)
      .setParam("severity", "INFO")).failIfNotSuccessful();
  }

}
