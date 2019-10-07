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
import org.sonar.python.api.tree.ComprehensionExpression;
import org.sonar.python.api.tree.ComprehensionFor;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class ComprehensionExpressionImpl extends PyTree implements ComprehensionExpression {

  private final Kind kind;
  private final Token openingToken;
  private final Expression resultExpression;
  private final ComprehensionFor comprehensionFor;
  private final Token closingToken;

  public ComprehensionExpressionImpl(Kind kind, Token openingToken, Expression resultExpression,
                                     ComprehensionFor compFor, Token closingToken) {
    super(openingToken, closingToken);
    this.kind = kind;
    this.resultExpression = resultExpression;
    this.comprehensionFor = compFor;
    this.openingToken = openingToken;
    this.closingToken = closingToken;
  }

  @Override
  public Expression resultExpression() {
    return resultExpression;
  }

  @Override
  public ComprehensionFor comprehensionFor() {
    return comprehensionFor;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitPyListOrSetCompExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(openingToken, resultExpression, comprehensionFor, closingToken).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return kind;
  }
}
