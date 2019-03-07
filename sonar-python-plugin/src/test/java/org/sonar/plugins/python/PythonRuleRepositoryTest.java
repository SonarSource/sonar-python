/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.List;
import org.junit.Test;
import org.sonar.api.SonarRuntime;
import org.sonar.api.rules.RuleType;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.python.checks.CheckList;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonRuleRepositoryTest {

  @Test
  public void createRulesTest() {
    RulesDefinition.Repository repository = buildRepository();

    assertThat(repository).isNotNull();
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.name()).isEqualTo("SonarAnalyzer");

    List<RulesDefinition.Rule> rules = repository.rules();
    assertThat(rules).isNotNull();
    assertThat(rules).hasSize(64);

    RulesDefinition.Rule s1578 = repository.rule("S1578");
    assertThat(s1578).isNotNull();
    assertThat(s1578.activatedByDefault()).isFalse();
    RulesDefinition.Rule backstickUsage = repository.rule("BackticksUsage");
    assertThat(backstickUsage).isNotNull();
    assertThat(backstickUsage.activatedByDefault()).isTrue();
  }

  @Test
  public void ruleTemplates() {
    RulesDefinition.Repository repository = buildRepository();
    assertThat(repository).isNotNull();

    RulesDefinition.Rule rule;

    rule = repository.rule("S100");
    assertThat(rule).isNotNull();
    assertThat(rule.template()).isFalse();

    rule = repository.rule("CommentRegularExpression");
    assertThat(rule).isNotNull();
    assertThat(rule.template()).isTrue();

    rule = repository.rule("XPath");
    assertThat(rule).isNotNull();
    assertThat(rule.template()).isTrue();

    long templateCount = repository.rules().stream()
      .map(RulesDefinition.Rule::template)
      .filter(Boolean::booleanValue)
      .count();
    assertThat(repository.rules().size()).isGreaterThan(50);
    assertThat(templateCount).isEqualTo(2);
  }

  @Test
  public void hotspotRules() {
    RulesDefinition.Repository repository = buildRepository();
    RulesDefinition.Rule hardcodedIp = repository.rule("S1313");
    assertThat(hardcodedIp.type()).isEqualTo(RuleType.SECURITY_HOTSPOT);
  }

  @Test
  public void hotspotRulesNotSupported() {
    RulesDefinition.Repository repository = buildRepository(TestUtils.SONAR_RUNTIME_67);
    RulesDefinition.Rule hardcodedIp = repository.rule("S1313");
    assertThat(hardcodedIp.type()).isEqualTo(RuleType.VULNERABILITY);
  }

  private static RulesDefinition.Repository buildRepository() {
    return buildRepository(TestUtils.SONAR_RUNTIME_72);
  }

  private static RulesDefinition.Repository buildRepository(SonarRuntime sonarRuntime) {
    PythonRuleRepository ruleRepository = new PythonRuleRepository(sonarRuntime);
    RulesDefinition.Context context = new RulesDefinition.Context();
    ruleRepository.define(context);
    return context.repository(CheckList.REPOSITORY_KEY);
  }

}
