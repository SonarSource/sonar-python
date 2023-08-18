/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class StringElementImplTest {

  @Test
  void isInterpolated() {
    assertThat(stringElement("'abc'").isInterpolated()).isFalse();
    assertThat(stringElement("f'abc'").isInterpolated()).isTrue();
    assertThat(stringElement("F'abc'").isInterpolated()).isTrue();
    assertThat(stringElement("r'abc'").isInterpolated()).isFalse();
    assertThat(stringElement("rf'abc'").isInterpolated()).isTrue();
  }

  private StringElement stringElement(String code) {
    return ((StringLiteral) PythonTestUtils.lastExpression(code)).stringElements().get(0);
  }
}
