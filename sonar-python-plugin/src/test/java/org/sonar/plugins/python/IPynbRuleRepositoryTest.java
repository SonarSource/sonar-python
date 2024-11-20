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
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.utils.Version;
import org.sonar.python.checks.CheckList;

import static org.assertj.core.api.Assertions.assertThat;

class IPynbRuleRepositoryTest {

  @Test
  void createRulesTest() throws IOException {
    RulesDefinition.Repository repository = buildRepository();

    assertThat(repository).isNotNull();
    assertThat(repository.language()).isEqualTo("ipynb");
    assertThat(repository.name()).isEqualTo("Sonar");

    List<RulesDefinition.Rule> rules = repository.rules();
    assertThat(rules)
      .isNotNull()
      .hasSameSizeAs(nonAbstractCheckFiles());

    RulesDefinition.Rule s1578 = repository.rule("S1578");
    assertThat(s1578).isNotNull();
    assertThat(s1578.activatedByDefault()).isFalse();
    RulesDefinition.Rule backstickUsage = repository.rule("BackticksUsage");
    assertThat(backstickUsage).isNotNull();
    assertThat(backstickUsage.activatedByDefault()).isTrue();

    RulesDefinition.Rule s905 = repository.rule("S905");
    assertThat(s905).isNotNull();
    assertThat(s905.activatedByDefault()).isFalse();

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
      .toList();
  }

  private static RulesDefinition.Repository buildRepository() {
    return buildRepository(9, 3);
  }

  private static RulesDefinition.Repository buildRepository(int majorVersion, int minorVersion) {
    SonarRuntime sonarRuntime = SonarRuntimeImpl.forSonarQube(Version.create(majorVersion, minorVersion), SonarQubeSide.SERVER, SonarEdition.DEVELOPER);
    IPynbRuleRepository ruleRepository = new IPynbRuleRepository(sonarRuntime);
    RulesDefinition.Context context = new RulesDefinition.Context();
    ruleRepository.define(context);
    return context.repository(CheckList.IPYTHON_REPOSITORY_KEY);
  }
}
