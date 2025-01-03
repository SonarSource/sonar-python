/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.plugins.python.mypy;

import org.junit.jupiter.api.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

class MypyRulesDefinitionTest {

  @Test
  void mypy_external_repository() {
    RulesDefinition.Context context = new RulesDefinition.Context();
    MypyRulesDefinition rulesDefinition = new MypyRulesDefinition();
    rulesDefinition.define(context);

    assertThat(context.repositories()).hasSize(1);
    RulesDefinition.Repository repository = context.repository("external_mypy");
    assertThat(repository).isNotNull();
    assertThat(repository.name()).isEqualTo("Mypy");
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.isExternal()).isTrue();
    assertThat(repository.rules()).hasSize(52);

    RulesDefinition.Rule rule = repository.rule("attr-defined");
    assertThat(rule).isNotNull();
    assertThat(rule.name()).isEqualTo("Check that attribute exists");
    assertThat(rule.htmlDescription()).isEqualTo("See description of Mypy rule <code>attr-defined</code> at" +
      " the <a href=\"https://mypy.readthedocs.io/en/stable/error_code_list.html#check-that-attribute-exists-attr-defined\">Mypy website</a>.");
    assertThat(rule.tags()).isEmpty();
  }
}
