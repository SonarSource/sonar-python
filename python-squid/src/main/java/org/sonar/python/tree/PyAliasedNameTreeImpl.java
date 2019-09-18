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
import java.util.Arrays;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyAliasedNameTreeImpl extends PyTree implements PyAliasedNameTree {

  private final PyToken asKeyword;
  private final PyDottedNameTree dottedName;
  private final PyNameTree alias;

  public PyAliasedNameTreeImpl(AstNode astNode, @Nullable PyToken asKeyword, PyDottedNameTree dottedName, @Nullable PyNameTree alias) {
    super(astNode);
    this.asKeyword = asKeyword;
    this.dottedName = dottedName;
    this.alias = alias;
  }

  @CheckForNull
  @Override
  public PyToken asKeyword() {
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

  @Override
  public List<Tree> children() {
    return Arrays.asList(dottedName, alias);
  }
}
