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
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TupleParameter;

public class TupleParameterImpl extends PyTree implements TupleParameter {

  private final Token lParenthesis;
  private final List<AnyParameter> parameters;
  private final List<Token> commas;
  private final Token rParenthesis;

  public TupleParameterImpl(Token lParenthesis, List<AnyParameter> parameters, List<Token> commas, Token rParenthesis) {
    this.lParenthesis = lParenthesis;
    this.parameters = parameters;
    this.commas = commas;
    this.rParenthesis = rParenthesis;
  }

  @Override
  public Token openingParenthesis() {
    return lParenthesis;
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
    return rParenthesis;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTupleParameter(this);
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    children.add(lParenthesis);
    int i = 0;
    for (Tree argument : parameters) {
      children.add(argument);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    children.add(rParenthesis);
    return children;
  }

  @Override
  public Kind getKind() {
    return Kind.TUPLE_PARAMETER;
  }
}
