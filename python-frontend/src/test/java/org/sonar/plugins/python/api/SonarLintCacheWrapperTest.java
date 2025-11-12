/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.api;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

class SonarLintCacheWrapperTest {
  @Test
  void testEmptyConstructor() {
    SonarLintCacheWrapper wrapper = new SonarLintCacheWrapper();
    assertThat(wrapper.sonarLintCache()).isNull();
  }

  @Test
  void testConstructorWithParameter() {
    SonarLintCacheWrapper wrapper = new SonarLintCacheWrapper(new SonarLintCache());
    assertThat(wrapper.sonarLintCache()).isNotNull();
  }
}
