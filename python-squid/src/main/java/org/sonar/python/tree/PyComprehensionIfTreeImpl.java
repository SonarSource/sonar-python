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
import org.sonar.python.api.tree.PyComprehensionClauseTree;
import org.sonar.python.api.tree.PyComprehensionIfTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyComprehensionIfTreeImpl extends PyTree implements PyComprehensionIfTree {

  private final PyToken ifToken;
  private final PyExpressionTree condition;
  private final PyComprehensionClauseTree nestedClause;

  public PyComprehensionIfTreeImpl(AstNode node, PyToken ifToken, PyExpressionTree condition, @Nullable PyComprehensionClauseTree nestedClause) {
    super(node);
    this.ifToken = ifToken;
    this.condition = condition;
    this.nestedClause = nestedClause;
  }

  @Override
  public PyToken ifToken() {
    return ifToken;
  }

  @Override
  public PyExpressionTree condition() {
    return condition;
  }

  @CheckForNull
  @Override
  public PyComprehensionClauseTree nestedClause() {
    return nestedClause;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitComprehensionIf(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(condition, nestedClause);
  }

  @Override
  public Kind getKind() {
    return Kind.COMP_IF;
  }
}
