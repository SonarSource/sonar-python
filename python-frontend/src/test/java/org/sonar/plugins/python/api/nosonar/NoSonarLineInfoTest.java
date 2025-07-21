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
package org.sonar.plugins.python.api.nosonar;

import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

class NoSonarLineInfoTest {

  @Test
  void constructorWithDefaultValueTest() {
    NoSonarLineInfo noSonarLineInfo = new NoSonarLineInfo(Set.of());
    Assertions.assertThat(noSonarLineInfo.comment()).isEmpty();
  }

  @Test
  void isSuppressedRuleKeysEmptyTest() {
    NoSonarLineInfo noSonarLineInfoEmpty = new NoSonarLineInfo(Set.of());
    Assertions.assertThat(noSonarLineInfoEmpty.isSuppressedRuleKeysEmpty()).isTrue();

    NoSonarLineInfo noSonarLineInfoNotEmpty = new NoSonarLineInfo(Set.of("rule1", "rule2"));
    Assertions.assertThat(noSonarLineInfoNotEmpty.isSuppressedRuleKeysEmpty()).isFalse();

  }
}
