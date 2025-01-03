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
package org.sonar.python.tree;

import org.junit.jupiter.api.Test;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.types.InferredTypes.BOOL;

class UnaryExpressionImplTest {

  @Test
  void type() {
    assertThat(lastExpression("not 42").type()).isEqualTo(BOOL);
    assertThat(lastExpression("-42").type()).isEqualTo(InferredTypes.INT);
    assertThat(lastExpression("+42").type()).isEqualTo(InferredTypes.INT);
    assertThat(lastExpression("+4.2").type()).isEqualTo(InferredTypes.FLOAT);
    assertThat(lastExpression("~42").type()).isEqualTo(InferredTypes.anyType());
  }
}
