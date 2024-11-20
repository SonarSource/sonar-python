/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.isDeclaredTypeWithTypeClass;

class SliceExpressionImplTest {

  @Test
  void type() {
    assertThat(lastExpression("[42, 43][1:2]").type()).isEqualTo(LIST);
    assertThat(lastExpression("(42, 43)[1:2]").type()).isEqualTo(TUPLE);
    assertThat(lastExpression("foo()[1:2]").type()).isEqualTo(InferredTypes.anyType());

    assertThat(isDeclaredTypeWithTypeClass(lastExpression(
      "from typing import List",
      "def f(x: List[int]): x[1:2]"
    ).type(), BuiltinTypes.LIST)).isTrue();

    assertThat(isDeclaredTypeWithTypeClass(lastExpression(
      "from typing import Tuple",
      "def f(x: Tuple[int]): x[1:2]"
    ).type(), BuiltinTypes.TUPLE)).isTrue();

    assertThat(lastExpression(
      "from typing import Set",
      "def f(x: Set[int]): x[1:2]"
    ).type()).isEqualTo(anyType());
  }

  @Test
  void type_dependencies() {
    SliceExpressionImpl sliceExpression = (SliceExpressionImpl) lastExpression("[42, 43][1:2]");
    assertThat(sliceExpression.typeDependencies()).containsExactly(sliceExpression.object());
  }
}
