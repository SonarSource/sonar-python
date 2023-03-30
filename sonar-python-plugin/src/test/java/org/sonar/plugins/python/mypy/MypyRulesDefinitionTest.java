/*
 * Copyright (C) 2021-2023 SonarSource SA
 * All rights reserved
 * mailto:info AT sonarsource DOT com
 */
package org.sonar.plugins.python.mypy;

import org.junit.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

public class MypyRulesDefinitionTest {

  @Test
  public void mypy_external_repository() {
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
