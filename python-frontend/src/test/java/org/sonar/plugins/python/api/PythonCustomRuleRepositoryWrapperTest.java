/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.AssertionsForClassTypes.assertThat;

class PythonCustomRuleRepositoryWrapperTest {
  @Test
  void testEmptyConstructor() {
    PythonCustomRuleRepositoryWrapper wrapper = new PythonCustomRuleRepositoryWrapper();
    assertThat(wrapper.customRuleRepositories()).isNotNull();
    assertThat(wrapper.customRuleRepositories()).isEmpty();
  }

  @Test
  void testConstructorWithNull() {
    PythonCustomRuleRepositoryWrapper wrapper = new PythonCustomRuleRepositoryWrapper(null);
    assertThat(wrapper.customRuleRepositories()).isNotNull();
    assertThat(wrapper.customRuleRepositories()).isEmpty();
  }
}
