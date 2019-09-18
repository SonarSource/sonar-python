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
import org.sonar.python.api.tree.PyToken;
import java.util.Arrays;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyQualifiedExpressionTreeImpl extends PyTree implements PyQualifiedExpressionTree {
  private final PyNameTree name;
  private final PyExpressionTree qualifier;
  private final PyToken dotToken;
  private final AstNode astNode;

  public PyQualifiedExpressionTreeImpl(AstNode astNode, PyNameTree name, PyExpressionTree qualifier, PyToken dotToken) {
    super(qualifier.firstToken(), name.lastToken());
    // FIXME : astNode is required to make semantic work at function level, this should disappear once semantic relies on strongly typed AST.
    this.astNode = astNode;
    this.name = name;
    this.qualifier = qualifier;
    this.dotToken = dotToken;
  }

  @CheckForNull
  @Override
  public AstNode astNode() {
    return astNode;
  }

  @Override
  public PyExpressionTree qualifier() {
    return qualifier;
  }

  @Override
  public PyToken dotToken() {
    return dotToken;
  }

  @Override
  public PyNameTree name() {
    return name;
  }

  @Override
  public Kind getKind() {
    return Kind.QUALIFIED_EXPR;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitQualifiedExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(name, qualifier);
  }
}
