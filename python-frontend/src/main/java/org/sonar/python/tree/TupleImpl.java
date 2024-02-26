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

import java.util.ArrayList;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

public class TupleImpl extends PyTree implements Tuple {

  private final Token leftParenthesis;
  private final List<Expression> elements;
  private final List<Token> commas;
  private final Token rightParenthesis;

  public TupleImpl(@Nullable Token leftParenthesis, List<Expression> elements, List<Token> commas, @Nullable Token rightParenthesis) {
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
  public List<Expression> elements() {
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
  public void accept(TreeVisitor visitor) {
    visitor.visitTuple(this);
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    if (leftParenthesis != null) {
      children.add(leftParenthesis);
    }
    int i = 0;
    for (Tree element : elements) {
      children.add(element);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    if (rightParenthesis != null) {
      children.add(rightParenthesis);
    }
    return children;
  }

  @Override
  public Kind getKind() {
    return Kind.TUPLE;
  }

  @Override
  public InferredType type() {
    return InferredTypes.TUPLE;
  }
}
