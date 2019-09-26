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

import com.sonar.sslr.api.AstNode;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.Token;
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.ReturnStatement;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class ReturnStatementImpl extends PyTree implements ReturnStatement {
  private final Token returnKeyword;
  private final List<Expression> expressionTrees;

  public ReturnStatementImpl(AstNode astNode, Token returnKeyword, List<Expression> expressionTrees) {
    super(astNode);
    this.returnKeyword = returnKeyword;
    this.expressionTrees = expressionTrees;
  }

  @Override
  public Token returnKeyword() {
    return returnKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressionTrees;
  }

  @Override
  public Kind getKind() {
    return Kind.RETURN_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitReturnStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(returnKeyword), expressionTrees).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
