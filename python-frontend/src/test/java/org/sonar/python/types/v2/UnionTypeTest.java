/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.types.v2;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;

import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;
import static org.assertj.core.api.Assertions.assertThat;


class UnionTypeTest {

  @Test
  void basicUnion() {
    FileInput fileInput = parseAndInferTypes("42;\"hello\"");
    PythonType intType = ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2();
    PythonType strType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();

    UnionType unionType = new UnionType(List.of(intType, strType));

    assertThat(unionType.isCompatibleWith(intType)).isTrue();
    assertThat(unionType.isCompatibleWith(strType)).isTrue();

    assertThat(unionType.displayName()).contains("Union[int, str]");
    assertThat(unionType.instanceDisplayName()).isEmpty();
  }

  @Test
  void unionWithUnknown() {
    FileInput fileInput = parseAndInferTypes("42;foo()");
    PythonType intType = ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2();
    PythonType strType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    UnionType unionType = new UnionType(List.of(intType, strType));

    assertThat(unionType.displayName()).isEmpty();
    assertThat(unionType.instanceDisplayName()).isEmpty();
  }
}
