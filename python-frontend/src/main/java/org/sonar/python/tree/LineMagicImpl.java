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
import org.sonar.plugins.python.api.tree.LineMagic;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class LineMagicImpl extends PyTree implements LineMagic {

  private final Token percent;
  private final Tree name;
  private final List<Token> tokens;

  public LineMagicImpl(Token percent, Tree name, List<Token> tokens) {
    this.percent = percent;
    this.name = name;
    this.tokens = tokens;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    // no op
  }

  @Override
  public Kind getKind() {
    return Kind.LINE_MAGIC;
  }

  @Override
  List<Tree> computeChildren() {
    var children = new ArrayList<Tree>();
    children.add(percent);
    children.add(name);
    children.addAll(tokens);
    return children;
  }

}
