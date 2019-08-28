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
import com.sonar.sslr.api.Token;
import java.util.List;
import java.util.Arrays;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyArgumentTreeImpl extends PyTree implements PyArgumentTree {
  private final PyNameTree keywordArgument;
  private final PyExpressionTree expression;
  private final Token equalToken;
  private final Token star;
  private final Token starStar;

  public PyArgumentTreeImpl(AstNode node, PyNameTree keywordArgument, PyExpressionTree expression, Token equalToken, AstNode star, AstNode starStar) {
    super(node);
    this.keywordArgument = keywordArgument;
    this.expression = expression;
    this.equalToken = equalToken;
    this.star = star != null ? star.getToken() : null;
    this.starStar = starStar != null ? starStar.getToken() : null;
  }

  public PyArgumentTreeImpl(AstNode astNode, PyExpressionTree expression, AstNode star, AstNode starStar) {
    super(astNode);
    this.keywordArgument = null;
    this.expression = expression;
    this.equalToken = null;
    this.star = star != null ? star.getToken() : null;
    this.starStar = starStar != null ? starStar.getToken() : null;
  }

  @CheckForNull
  @Override
  public PyNameTree keywordArgument() {
    return keywordArgument;
  }

  @CheckForNull
  @Override
  public Token equalToken() {
    return equalToken;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @CheckForNull
  @Override
  public Token starToken() {
    return star;
  }

  @CheckForNull
  @Override
  public Token starStarToken() {
    return starStar;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitArgument(this);
  }

  @Override
  public Kind getKind() {
    return Kind.ARGUMENT;
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(keywordArgument, expression);
  }
}
