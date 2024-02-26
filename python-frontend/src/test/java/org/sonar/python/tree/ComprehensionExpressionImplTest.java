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
package org.sonar.python.tree;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class ComprehensionExpressionImplTest {
  @Test
  void type() {
    Token token = mock(Token.class);
    Expression resultExpression = mock(Expression.class);
    ComprehensionFor compFor = mock(ComprehensionFor.class);

    ComprehensionExpressionImpl generator = new ComprehensionExpressionImpl(Kind.GENERATOR_EXPR, token, resultExpression, compFor, token);
    assertThat(generator.type()).isEqualTo(InferredTypes.anyType());

    ComprehensionExpressionImpl listComp = new ComprehensionExpressionImpl(Kind.LIST_COMPREHENSION, token, resultExpression, compFor, token);
    assertThat(listComp.type()).isEqualTo(InferredTypes.LIST);

    ComprehensionExpressionImpl setComp = new ComprehensionExpressionImpl(Kind.SET_COMPREHENSION, token, resultExpression, compFor, token);
    assertThat(setComp.type()).isEqualTo(InferredTypes.SET);

    // this will never happen, added for coverage
    ComprehensionExpressionImpl comprehensionExpression = new ComprehensionExpressionImpl(Kind.NAME, token, resultExpression, compFor, token);
    assertThat(comprehensionExpression.type()).isEqualTo(InferredTypes.anyType());
  }
}
