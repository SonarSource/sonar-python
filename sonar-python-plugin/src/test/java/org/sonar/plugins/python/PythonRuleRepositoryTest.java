/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.api.rules.RuleType;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.python.checks.CheckList;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonRuleRepositoryTest {

  @Test
  public void createRulesTest() throws IOException {
    RulesDefinition.Repository repository = buildRepository();

    assertThat(repository).isNotNull();
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.name()).isEqualTo("SonarQube");

    List<RulesDefinition.Rule> rules = repository.rules();
    assertThat(rules).isNotNull();
    assertThat(rules).hasSameSizeAs(nonAbstractCheckFiles());

    RulesDefinition.Rule s1578 = repository.rule("S1578");
    assertThat(s1578).isNotNull();
    assertThat(s1578.activatedByDefault()).isFalse();
    RulesDefinition.Rule backstickUsage = repository.rule("BackticksUsage");
    assertThat(backstickUsage).isNotNull();
    assertThat(backstickUsage.activatedByDefault()).isTrue();

    for (RulesDefinition.Rule rule : rules) {
      assertThat(rule.htmlDescription()).isNotEmpty();
      rule.params().forEach(p -> assertThat(p.description()).isNotEmpty());
    }
  }

  private List<String> nonAbstractCheckFiles() throws IOException {
    return Files.walk(new File("../python-checks/src/main/java/org/sonar/python/checks").toPath())
      .filter(Files::isRegularFile)
      .map(p -> p.getFileName().toString())
      .filter(f -> f.endsWith("Check.java"))
      .filter(f -> !f.startsWith("Abstract"))
      .collect(Collectors.toList());
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

    long templateCount = repository.rules().stream()
      .map(RulesDefinition.Rule::template)
      .filter(Boolean::booleanValue)
      .count();
    assertThat(repository.rules().size()).isGreaterThan(50);
    assertThat(templateCount).isEqualTo(1);
  }

  @Test
  public void hotspotRules() {
    RulesDefinition.Repository repository = buildRepository();
    RulesDefinition.Rule hardcodedIp = repository.rule("S1313");
    assertThat(hardcodedIp.type()).isEqualTo(RuleType.SECURITY_HOTSPOT);
  }

  private static RulesDefinition.Repository buildRepository() {
    PythonRuleRepository ruleRepository = new PythonRuleRepository();
    RulesDefinition.Context context = new RulesDefinition.Context();
    ruleRepository.define(context);
    return context.repository(CheckList.REPOSITORY_KEY);
  }

}
