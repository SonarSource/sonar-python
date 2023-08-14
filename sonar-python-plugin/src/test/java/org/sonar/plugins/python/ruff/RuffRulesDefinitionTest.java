/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.plugins.python.ruff;

import org.junit.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

public class RuffRulesDefinitionTest {

  @Test
  public void ruff_external_repository() {
    RulesDefinition.Context context = new RulesDefinition.Context();
    RuffRulesDefinition rulesDefinition = new RuffRulesDefinition();
    rulesDefinition.define(context);

    assertThat(context.repositories()).hasSize(1);
    RulesDefinition.Repository repository = context.repository("external_ruff");
    assertThat(repository).isNotNull();
    assertThat(repository.name()).isEqualTo("Ruff");
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.isExternal()).isEqualTo(true);
    assertThat(repository.rules().size()).isEqualTo(679);

    RulesDefinition.Rule rule = repository.rule("F405");
    assertThat(rule).isNotNull();
    assertThat(rule.name()).isEqualTo("undefined-local-with-import-star-usage");
    assertThat(rule.htmlDescription()).isEqualTo("See description of Ruff rule <code>F405</code> at" +
      " the <a href=\"https://beta.ruff.rs/docs/rules/undefined-local-with-import-star-usage\">Ruff website</a>.");
    assertThat(rule.tags()).isEmpty();
  }
}
