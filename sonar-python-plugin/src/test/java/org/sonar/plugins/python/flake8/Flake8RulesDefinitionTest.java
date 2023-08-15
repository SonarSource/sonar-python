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
package org.sonar.plugins.python.flake8;

import org.junit.jupiter.api.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

public class Flake8RulesDefinitionTest {

  @Test
  public void flake8_external_repository() {
    RulesDefinition.Context context = new RulesDefinition.Context();
    Flake8RulesDefinition rulesDefinition = new Flake8RulesDefinition();
    rulesDefinition.define(context);

    assertThat(context.repositories()).hasSize(1);
    RulesDefinition.Repository repository = context.repository("external_flake8");
    assertThat(repository).isNotNull();
    assertThat(repository.name()).isEqualTo("Flake8");
    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.isExternal()).isEqualTo(true);
    assertThat(repository.rules().size()).isEqualTo(116);

    RulesDefinition.Rule rule = repository.rule("F405");
    assertThat(rule).isNotNull();
    assertThat(rule.name()).isEqualTo("name may be undefined, or defined from star imports: module");
    assertThat(rule.htmlDescription()).isEqualTo("See description of Flake8 rule <code>F405</code> at" +
      " the <a href=\"https://flake8.pycqa.org/en/latest/user/error-codes.html\">Flake8 website</a>.");
    assertThat(rule.tags()).isEmpty();
  }
}
