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
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.PyGlobalStatementTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyGlobalStatementTreeImpl extends PyTree implements PyGlobalStatementTree {
  private final PyToken globalKeyword;
  private final List<PyNameTree> variables;

  public PyGlobalStatementTreeImpl(AstNode astNode, PyToken globalKeyword, List<PyNameTree> variables) {
    super(astNode);
    this.globalKeyword = globalKeyword;
    this.variables = variables;
  }

  @Override
  public PyToken globalKeyword() {
    return globalKeyword;
  }

  @Override
  public List<PyNameTree> variables() {
    return variables;
  }

  @Override
  public Kind getKind() {
    return Kind.GLOBAL_STMT;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitGlobalStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.unmodifiableList(variables);
  }
}
