/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import org.sonar.plugins.python.api.tree.CellMagicStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;


public class CellMagicStatementImpl extends PyTree implements CellMagicStatement {

  private final List<Tree> tokens;

  public CellMagicStatementImpl(List<Tree> tokens) {
    this.tokens = tokens;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitCellMagicStatement(this);
  }

  @Override
  public Kind getKind() {
    return Kind.CELL_MAGIC_STATEMENT;
  }

  @Override
  List<Tree> computeChildren() {
    return new ArrayList<>(tokens);
  }
}
