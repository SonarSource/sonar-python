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
package org.sonar.plugins.python.bandit;

import org.junit.jupiter.api.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

class BanditRulesDefinitionTest {

  @Test
  void bandit_lint_external_repository() {
    RulesDefinition.Context context = new RulesDefinition.Context();
    BanditRulesDefinition rulesDefinition = new BanditRulesDefinition();
    rulesDefinition.define(context);

    assertThat(context.repositories()).hasSize(1);
    RulesDefinition.Repository repository = context.repository("external_bandit");
    assertThat(repository).isNotNull();
    assertThat(repository.name()).isEqualTo("Bandit");
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.isExternal()).isEqualTo(true);
    assertThat(repository.rules().size()).isEqualTo(73);

    RulesDefinition.Rule rule = repository.rule("B101");
    assertThat(rule).isNotNull();
    assertThat(rule.name()).isEqualTo("B101: Test for use of assert");
    assertThat(rule.htmlDescription()).isEqualTo("See description of Bandit rule <code>B101</code> at" +
      " the <a href=\"https://bandit.readthedocs.io/en/latest/plugins/b101_assert_used.html\">Bandit website</a>.");
    assertThat(rule.tags()).isEmpty();
  }

}
