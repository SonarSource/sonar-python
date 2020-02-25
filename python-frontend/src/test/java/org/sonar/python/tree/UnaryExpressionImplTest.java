/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class UnaryExpressionImplTest {

  @Test
  public void type() {
    Expression expression = mock(Expression.class);

    Token not = mock(Token.class);
    when(not.value()).thenReturn("not");
    assertThat(((UnaryExpression) new UnaryExpressionImpl(not, expression)).type()).isEqualTo(InferredTypes.BOOL);

    Token minus = mock(Token.class);
    when(minus.value()).thenReturn("-");
    assertThat(((UnaryExpression) new UnaryExpressionImpl(minus, expression)).type()).isEqualTo(InferredTypes.anyType());

    Token plus = mock(Token.class);
    when(plus.value()).thenReturn("+");
    assertThat(((UnaryExpression) new UnaryExpressionImpl(plus, expression)).type()).isEqualTo(InferredTypes.anyType());

    Token bitwiseComplement = mock(Token.class);
    when(bitwiseComplement.value()).thenReturn("~");
    assertThat(((UnaryExpression) new UnaryExpressionImpl(bitwiseComplement, expression)).type()).isEqualTo(InferredTypes.anyType());
  }
}
