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
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.AnyParameter;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.TupleParameter;
import org.sonar.python.api.tree.Tree;

public class TupleParameterImpl extends PyTree implements TupleParameter {

  private final List<AnyParameter> parameters;
  private final List<Token> commas;

  public TupleParameterImpl(AstNode node, List<AnyParameter> parameters, List<Token> commas) {
    super(node);
    this.parameters = parameters;
    this.commas = commas;
  }

  @Override
  public Token openingParenthesis() {
    return firstToken();
  }

  @Override
  public List<AnyParameter> parameters() {
    return parameters;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }

  @Override
  public Token closingParenthesis() {
    return lastToken();
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTupleParameter(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(parameters, commas).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.TUPLE_PARAMETER;
  }
}
