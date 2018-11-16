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
package org.sonar.plugins.python;

import java.util.List;
import org.junit.Test;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.python.checks.CheckList;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonRuleRepositoryTest {

  @Test
  public void createRulesTest() {
    RulesDefinition.Repository repository = buildRepository();

    assertThat(repository.language()).isEqualTo("py");
    assertThat(repository.name()).isEqualTo("SonarAnalyzer");

    List<RulesDefinition.Rule> rules = repository.rules();
    assertThat(rules).isNotNull();
    assertThat(rules).hasSize(54);

    assertThat(repository.rule("S1578").activatedByDefault()).isFalse();
    assertThat(repository.rule("BackticksUsage").activatedByDefault()).isTrue();
  }

  @Test
  public void ruleTemplates() {
    RulesDefinition.Repository repository = buildRepository();
    assertThat(repository.rule("S100").template()).isFalse();
    assertThat(repository.rule("CommentRegularExpression").template()).isTrue();
    assertThat(repository.rule("XPath").template()).isTrue();

    long templateCount = repository.rules().stream()
      .map(RulesDefinition.Rule::template)
      .filter(Boolean::booleanValue)
      .count();
    assertThat(repository.rules().size()).isGreaterThan(50);
    assertThat(templateCount).isEqualTo(2);
  }

  private RulesDefinition.Repository buildRepository() {
    PythonRuleRepository ruleRepository = new PythonRuleRepository();
    RulesDefinition.Context context = new RulesDefinition.Context();
    ruleRepository.define(context);
    return context.repository(CheckList.REPOSITORY_KEY);
  }

}
