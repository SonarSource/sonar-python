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
import java.util.Collections;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.Tree;

public class PyTupleTreeImpl extends PyTree implements PyTupleTree {

  private final Token leftParenthesis;
  private final List<PyExpressionTree> elements;
  private final List<Token> commas;
  private final Token rightParenthesis;

  public PyTupleTreeImpl(AstNode node, @Nullable Token leftParenthesis, List<PyExpressionTree> elements, List<Token> commas, @Nullable Token rightParenthesis) {
    super(node);
    this.leftParenthesis = leftParenthesis;
    this.elements = elements;
    this.commas = commas;
    this.rightParenthesis = rightParenthesis;
  }

  @CheckForNull
  @Override
  public Token leftParenthesis() {
    return leftParenthesis;
  }

  @Override
  public List<PyExpressionTree> elements() {
    return elements;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }

  @CheckForNull
  @Override
  public Token rightParenthesis() {
    return rightParenthesis;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitTuple(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.unmodifiableList(elements);
  }

  @Override
  public Kind getKind() {
    return Kind.TUPLE;
  }
}
