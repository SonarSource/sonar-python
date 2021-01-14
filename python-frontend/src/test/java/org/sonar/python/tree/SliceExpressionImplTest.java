/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.tree;

import org.junit.Test;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.isDeclaredTypeWithTypeClass;

public class SliceExpressionImplTest {

  @Test
  public void type() {
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
  public void type_dependencies() {
    SliceExpressionImpl sliceExpression = (SliceExpressionImpl) lastExpression("[42, 43][1:2]");
    assertThat(sliceExpression.typeDependencies()).containsExactly(sliceExpression.object());
  }
}
