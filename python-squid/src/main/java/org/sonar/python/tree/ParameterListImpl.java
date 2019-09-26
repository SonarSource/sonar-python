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
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.python.api.tree.AnyParameter;
import org.sonar.python.api.tree.Parameter;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.ParameterList;
import org.sonar.python.api.tree.Tree;

public class ParameterListImpl extends PyTree implements ParameterList {

  private final List<AnyParameter> parameters;

  public ParameterListImpl(AstNode node, List<AnyParameter> parameters) {
    super(node);
    this.parameters = parameters;
  }

  @Override
  public List<Parameter> nonTuple() {
    return parameters.stream()
      .filter(Parameter.class::isInstance)
      .map(Parameter.class::cast)
      .collect(Collectors.toList());
  }

  @Override
  public List<AnyParameter> all() {
    return Collections.unmodifiableList(parameters);
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitParameterList(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.unmodifiableList(parameters);
  }

  @Override
  public Kind getKind() {
    return Kind.PARAMETER_LIST;
  }
}
