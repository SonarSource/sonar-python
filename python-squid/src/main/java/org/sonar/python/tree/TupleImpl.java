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
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tuple;
import org.sonar.python.api.tree.Tree;

public class TupleImpl extends PyTree implements Tuple {

  private final Token leftParenthesis;
  private final List<Expression> elements;
  private final List<Token> commas;
  private final Token rightParenthesis;

  public TupleImpl(AstNode node, @Nullable Token leftParenthesis, List<Expression> elements, List<Token> commas, @Nullable Token rightParenthesis) {
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
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(leftParenthesis), elements, commas, Collections.singletonList(rightParenthesis))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.TUPLE;
  }
}
