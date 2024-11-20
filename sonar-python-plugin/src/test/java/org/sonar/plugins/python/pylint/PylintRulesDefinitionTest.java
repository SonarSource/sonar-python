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
package org.sonar.plugins.python.pylint;

import org.junit.jupiter.api.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

class PylintRulesDefinitionTest {

  @Test
  void pylint_external_repository() {
    RulesDefinition.Context context = new RulesDefinition.Context();
    PylintRulesDefinition rulesDefinition = new PylintRulesDefinition();
    rulesDefinition.define(context);

    assertThat(context.repositories()).hasSize(1);
    RulesDefinition.Repository repository = context.repository("external_pylint");
    assertThat(repository).isNotNull();
    assertThat(repository.name()).isEqualTo("Pylint");
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.isExternal()).isTrue();
    assertThat(repository.rules().size()).isEqualTo(370);

    RulesDefinition.Rule rule = repository.rule("C0121");
    assertThat(rule).isNotNull();
    assertThat(rule.name()).isEqualTo("Singleton comparison");
    assertThat(rule.htmlDescription()).isEqualTo("See description of Pylint rule <code>C0121</code> at " +
      "the <a href=\"https://pylint.pycqa.org/en/latest/technical_reference/features.html\">Pylint website</a>.");
    assertThat(rule.debtRemediationFunction().type().name()).isEqualTo("CONSTANT_ISSUE");
    assertThat(rule.debtRemediationFunction().baseEffort()).isEqualTo("5min");
    assertThat(rule.tags()).isEmpty();
  }
}
