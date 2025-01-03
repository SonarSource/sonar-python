/*
 * Copyright (C) 2011-2025 SonarSource SA - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python;

import org.junit.jupiter.api.Test;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.SonarRuntime;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.api.utils.Version;

import static org.assertj.core.api.Assertions.assertThat;

class CustomPythonRuleRepositoryTest {

  @Test
  void test_rule_repository() {
    SonarRuntime sonarRuntime = SonarRuntimeImpl.forSonarQube(Version.create(9, 9), SonarQubeSide.SCANNER, SonarEdition.DEVELOPER);
    CustomPythonRuleRepository customPythonRuleRepository = new CustomPythonRuleRepository(sonarRuntime);
    RulesDefinition.Context context = new RulesDefinition.Context();
    customPythonRuleRepository.define(context);
    assertThat(customPythonRuleRepository.repositoryKey()).isEqualTo("python-custom-rules-example");
    assertThat(context.repositories()).hasSize(1).extracting("key").containsExactly(customPythonRuleRepository.repositoryKey());
    var rules = context.repositories().get(0).rules();
    assertThat(rules).hasSize(2);
    assertThat(customPythonRuleRepository.checkClasses()).hasSize(2);
  }
}
