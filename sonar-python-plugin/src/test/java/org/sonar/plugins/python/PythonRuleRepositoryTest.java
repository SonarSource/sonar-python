/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.SonarRuntime;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.rules.RuleType;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.utils.Version;
import org.sonar.python.checks.CheckList;

import static org.assertj.core.api.Assertions.assertThat;

class PythonRuleRepositoryTest {

  @Test
  void createRulesTest() throws IOException {
    RulesDefinition.Repository repository = buildRepository();

    assertThat(repository).isNotNull();
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.name()).isEqualTo("Sonar");

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

  @Test
  void owaspSecurityStandard() {
    RulesDefinition.Repository repository_9_3 = buildRepository(9, 3);
    RulesDefinition.Rule s4721_9_3 = repository_9_3.rule("S4721");
    assertThat(s4721_9_3).isNotNull();
    assertThat(s4721_9_3.securityStandards()).contains("owaspTop10-2021:a3");

    RulesDefinition.Repository repository_9_2 = buildRepository(9, 2);
    RulesDefinition.Rule s4721_9_2 = repository_9_2.rule("S4721");
    assertThat(s4721_9_2).isNotNull();
    assertThat(s4721_9_2.securityStandards()).doesNotContain("owaspTop10-2021:a3");
  }

  @Test
  void psiDssSecurityStandard() {
    RulesDefinition.Repository repository_9_5 = buildRepository(9, 5);
    RulesDefinition.Rule s4792_9_5 = repository_9_5.rule("S4792");
    assertThat(s4792_9_5).isNotNull();
    assertThat(s4792_9_5.securityStandards()).contains("pciDss-3.2:10.1", "pciDss-3.2:10.2", "pciDss-3.2:10.3", "pciDss-4.0:10.2");

    RulesDefinition.Repository repository_9_4 = buildRepository(9, 4);
    RulesDefinition.Rule s4792_9_4 = repository_9_4.rule("S4792");
    assertThat(s4792_9_4).isNotNull();
    assertThat(s4792_9_4.securityStandards()).doesNotContain("pciDss-3.2:10.1", "pciDss-3.2:10.2", "pciDss-3.2:10.3", "pciDss-4.0:10.2");
  }

  private List<String> nonAbstractCheckFiles() throws IOException {
    return Files.walk(new File("../python-checks/src/main/java/org/sonar/python/checks").toPath())
      .filter(Files::isRegularFile)
      .map(p -> p.getFileName().toString())
      .filter(f -> f.endsWith("Check.java"))
      .filter(f -> !f.startsWith("Abstract"))
      .toList();
  }

  @Test
  void ruleTemplates() {
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
  void hotspotRules() {
    RulesDefinition.Repository repository = buildRepository();
    RulesDefinition.Rule hardcodedIp = repository.rule("S1313");
    assertThat(hardcodedIp.type()).isEqualTo(RuleType.SECURITY_HOTSPOT);
  }

  private static RulesDefinition.Repository buildRepository() {
    return buildRepository(9, 3);
  }

  private static RulesDefinition.Repository buildRepository(int majorVersion, int minorVersion) {
    SonarRuntime sonarRuntime = SonarRuntimeImpl.forSonarQube(Version.create(majorVersion, minorVersion), SonarQubeSide.SERVER, SonarEdition.DEVELOPER);
    PythonRuleRepository ruleRepository = new PythonRuleRepository(sonarRuntime);
    RulesDefinition.Context context = new RulesDefinition.Context();
    ruleRepository.define(context);
    return context.repository(CheckList.REPOSITORY_KEY);
  }

}
