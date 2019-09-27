/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.ComprehensionFor;
import org.sonar.python.api.tree.DictCompExpression;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class DictCompExpressionImpl extends PyTree implements DictCompExpression {

  private final Expression keyExpression;
  private final Token colon;
  private final Expression valueExpression;
  private final ComprehensionFor comprehensionFor;

  public DictCompExpressionImpl(Token openingBrace, Expression keyExpression, Token colon, Expression valueExpression,
                                ComprehensionFor compFor, Token closingBrace) {
    super(openingBrace, closingBrace);
    this.keyExpression = keyExpression;
    this.colon = colon;
    this.valueExpression = valueExpression;
    this.comprehensionFor = compFor;
  }

  @Override
  public Expression keyExpression() {
    return keyExpression;
  }

  @Override
  public Token colonToken() {
    return colon;
  }

  @Override
  public Expression valueExpression() {
    return valueExpression;
  }

  @Override
  public ComprehensionFor comprehensionFor() {
    return comprehensionFor;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDictCompExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(keyExpression, colon, valueExpression, comprehensionFor).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.DICT_COMPREHENSION;
  }
}
