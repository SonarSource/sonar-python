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
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyAliasedNameTreeImpl extends PyTree implements PyAliasedNameTree {

  private final Token asKeyword;
  private final PyDottedNameTree dottedName;
  private final PyNameTree alias;

  public PyAliasedNameTreeImpl(AstNode astNode, Token asKeyword, PyDottedNameTree dottedName, PyNameTree alias) {
    super(astNode);
    this.asKeyword = asKeyword;
    this.dottedName = dottedName;
    this.alias = alias;
  }

  @CheckForNull
  @Override
  public Token asKeyword() {
    return asKeyword;
  }

  @CheckForNull
  @Override
  public PyNameTree alias() {
    return alias;
  }

  @Override
  public PyDottedNameTree dottedName() {
    return dottedName;
  }

  @Override
  public Kind getKind() {
    return Kind.ALIASED_NAME;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitAliasedName(this);
  }
}
