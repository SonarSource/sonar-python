/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2025 SonarSource SA
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
package org.sonar.samples.python;

import org.junit.jupiter.api.Test;
import org.sonar.api.server.rule.RulesDefinition;

import static org.assertj.core.api.Assertions.assertThat;

class CustomPythonRuleRepositoryTest {

  @Test
  void test_rule_repository() {
    CustomPythonRuleRepository customPythonRuleRepository = new CustomPythonRuleRepository();
    RulesDefinition.Context context = new RulesDefinition.Context();
    customPythonRuleRepository.define(context);
    assertThat(customPythonRuleRepository.repositoryKey()).isEqualTo("python-custom-rules");
    assertThat(context.repositories()).hasSize(1).extracting("key").containsExactly(customPythonRuleRepository.repositoryKey());
    assertThat(context.repositories().get(0).rules()).hasSize(2);
    assertThat(customPythonRuleRepository.checkClasses()).hasSize(2);
  }
}
