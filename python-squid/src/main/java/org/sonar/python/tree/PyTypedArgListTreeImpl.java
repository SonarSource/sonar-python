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
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTypedArgListTree;
import org.sonar.python.api.tree.Tree;

public class PyTypedArgListTreeImpl extends PyTree implements PyTypedArgListTree {

  private final List<PyParameterTree> arguments;

  public PyTypedArgListTreeImpl(AstNode node, List<PyParameterTree> arguments) {
    super(node);
    this.arguments = arguments;
  }

  @Override
  public List<PyParameterTree> arguments() {
    return arguments;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitTypedArgList(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.unmodifiableList(arguments);
  }

  @Override
  public Kind getKind() {
    return Kind.TYPED_ARG_LIST;
  }
}
