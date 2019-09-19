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
import org.sonar.python.api.tree.PyToken;
import java.util.Collections;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.Tree;

public class PyTupleTreeImpl extends PyTree implements PyTupleTree {

  private final PyToken leftParenthesis;
  private final List<PyExpressionTree> elements;
  private final List<PyToken> commas;
  private final PyToken rightParenthesis;

  public PyTupleTreeImpl(AstNode node, @Nullable PyToken leftParenthesis, List<PyExpressionTree> elements, List<PyToken> commas, @Nullable PyToken rightParenthesis) {
    super(node);
    this.leftParenthesis = leftParenthesis;
    this.elements = elements;
    this.commas = commas;
    this.rightParenthesis = rightParenthesis;
  }

  @CheckForNull
  @Override
  public PyToken leftParenthesis() {
    return leftParenthesis;
  }

  @Override
  public List<PyExpressionTree> elements() {
    return elements;
  }

  @Override
  public List<PyToken> commas() {
    return commas;
  }

  @CheckForNull
  @Override
  public PyToken rightParenthesis() {
    return rightParenthesis;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitTuple(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(leftParenthesis), elements, commas, Collections.singletonList(rightParenthesis))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.TUPLE;
  }
}
