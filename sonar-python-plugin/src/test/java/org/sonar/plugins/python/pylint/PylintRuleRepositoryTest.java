/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import java.util.List;
import org.junit.Test;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.server.rule.RulesDefinitionXmlLoader;

import static org.assertj.core.api.Assertions.assertThat;

public class PylintRuleRepositoryTest {

  @Test
  public void createRulesTest() {
    PylintRuleRepository ruleRepository = new PylintRuleRepository(new RulesDefinitionXmlLoader());
    RulesDefinition.Context context = new RulesDefinition.Context();
    ruleRepository.define(context);

    RulesDefinition.Repository repository = context.repository(PylintRuleRepository.REPOSITORY_KEY);

    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.name()).isEqualTo("Pylint");

    List<RulesDefinition.Rule> rules = repository.rules();
    assertThat(rules).isNotNull();
    assertThat(rules).hasSize(322);

    long rulesWithoutRemediationCost = rules.stream()
      .filter(rule -> rule.debtRemediationFunction() == null)
      .count();
    assertThat(rulesWithoutRemediationCost).isEqualTo(28);
  }

}
