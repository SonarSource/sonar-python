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

import java.util.Collections;
import java.util.List;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.WildcardPattern;

public class WildcardPatternImpl extends PyTree implements WildcardPattern {
  private final Token wildcard;

  public WildcardPatternImpl(Token wildcard) {
    this.wildcard = wildcard;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitWildcardPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.WILDCARD_PATTERN;
  }

  @Override
  public Token wildcard() {
    return wildcard;
  }

  @Override
  List<Tree> computeChildren() {
    return Collections.singletonList(wildcard);
  }
}
