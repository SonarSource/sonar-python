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
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyFileInputTreeImpl extends PyTree implements PyFileInputTree {

  private final PyStatementListTree statements;
  private final Token docstring;

  public PyFileInputTreeImpl(AstNode astNode, @Nullable PyStatementListTree statements, Token docstring) {
    super(astNode);
    this.statements = statements;
    this.docstring = docstring;
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.FILE_INPUT;
  }

  @Override
  @CheckForNull
  public PyStatementListTree statements() {
    return statements;
  }

  @CheckForNull
  @Override
  public Token docstring() {
    return docstring;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitFileInput(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.singletonList(statements);
  }
}
