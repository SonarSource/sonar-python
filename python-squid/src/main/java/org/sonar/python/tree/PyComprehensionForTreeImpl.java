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
import java.util.Arrays;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyComprehensionClauseTree;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyComprehensionForTreeImpl extends PyTree implements PyComprehensionForTree {

  private final Token forToken;
  private final PyExpressionTree loopExpression;
  private final Token inToken;
  private final PyExpressionTree iterable;
  private final PyComprehensionClauseTree nested;

  public PyComprehensionForTreeImpl(AstNode node, Token forToken, PyExpressionTree loopExpression, Token inToken,
                                    PyExpressionTree iterable, @Nullable PyComprehensionClauseTree nested) {
    super(node);
    this.forToken = forToken;
    this.loopExpression = loopExpression;
    this.inToken = inToken;
    this.iterable = iterable;
    this.nested = nested;
  }

  @Override
  public Token forToken() {
    return forToken;
  }

  @Override
  public PyExpressionTree loopExpression() {
    return loopExpression;
  }

  @Override
  public Token inToken() {
    return inToken;
  }

  @Override
  public PyExpressionTree iterable() {
    return iterable;
  }

  @CheckForNull
  @Override
  public PyComprehensionClauseTree nestedClause() {
    return nested;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitComprehensionFor(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(loopExpression, iterable, nested);
  }

  @Override
  public Kind getKind() {
    return Kind.COMP_FOR;
  }
}
