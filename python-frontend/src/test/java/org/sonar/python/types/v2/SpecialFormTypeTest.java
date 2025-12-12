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
package org.sonar.python.types.v2;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class SpecialFormTypeTest {
  @Test
  void testEqualsAndHashCodeWithSameObject() {
    SpecialFormType specialFormType1 = new SpecialFormType("typing.Self");
    SpecialFormType specialFormType2 = new SpecialFormType("typing.Self");
    assertThat(specialFormType1)
      .isEqualTo(specialFormType2)
      .hasSameHashCodeAs(specialFormType2);
  }

  @Test
  void testEqualsAndHashCodeWithDifferentObject() {
    SpecialFormType specialFormType1 = new SpecialFormType("typing.Self");
    SpecialFormType specialFormType2 = new SpecialFormType("typing.Generic");
    assertThat(specialFormType1)
      .isNotEqualTo(specialFormType2)
      .doesNotHaveSameHashCodeAs(specialFormType2);
  }

  @Test
  void testEqualsWithDifferentClass() {
    SpecialFormType specialFormType1 = new SpecialFormType("typing.Self");
    assertThat(specialFormType1)
      .isNotEqualTo(new Object());
  }
}
