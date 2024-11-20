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
package org.sonar.plugins.python.ruff;

import org.junit.jupiter.api.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

class RuffRulesDefinitionTest {

  @Test
  void ruff_external_repository() {
    RulesDefinition.Context context = new RulesDefinition.Context();
    RuffRulesDefinition rulesDefinition = new RuffRulesDefinition();
    rulesDefinition.define(context);

    assertThat(context.repositories()).hasSize(1);
    RulesDefinition.Repository repository = context.repository("external_ruff");
    assertThat(repository).isNotNull();
    assertThat(repository.name()).isEqualTo("Ruff");
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.isExternal()).isTrue();
    assertThat(repository.rules()).hasSize(679);

    RulesDefinition.Rule rule = repository.rule("F405");
    assertThat(rule).isNotNull();
    assertThat(rule.name()).isEqualTo("undefined-local-with-import-star-usage");
    assertThat(rule.htmlDescription()).isEqualTo("See description of Ruff rule <code>F405</code> at" +
      " the <a href=\"https://beta.ruff.rs/docs/rules/undefined-local-with-import-star-usage\">Ruff website</a>.");
    assertThat(rule.tags()).isEmpty();

    RulesDefinition.Rule unknownRule = repository.rule("ZZZ123");
    assertThat(unknownRule).isNull();
  }
}
