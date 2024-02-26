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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.symbols.Symbol;

public class LambdaExpressionImpl extends PyTree implements LambdaExpression {
  private final Token lambdaKeyword;
  private final Token colonToken;
  private final Expression body;
  private final ParameterList parameterList;
  private Set<Symbol> symbols = new HashSet<>();

  public LambdaExpressionImpl(Token lambdaKeyword, Token colonToken, Expression body, @Nullable ParameterList parameterList) {
    this.lambdaKeyword = lambdaKeyword;
    this.colonToken = colonToken;
    this.body = body;
    this.parameterList = parameterList;
  }

  @Override
  public Token lambdaKeyword() {
    return lambdaKeyword;
  }

  @Override
  public Token colonToken() {
    return colonToken;
  }

  @Override
  public Expression expression() {
    return body;
  }

  @CheckForNull
  @Override
  public ParameterList parameters() {
    return parameterList;
  }

  @Override
  public Set<Symbol> localVariables() {
    return symbols;
  }

  @Override
  public boolean isMethodDefinition() {
    return false;
  }

  public void addLocalVariableSymbol(Symbol symbol) {
    symbols.add(symbol);
  }

  @Override
  public Kind getKind() {
    return Kind.LAMBDA;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitLambda(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(lambdaKeyword, parameterList, colonToken, body).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
