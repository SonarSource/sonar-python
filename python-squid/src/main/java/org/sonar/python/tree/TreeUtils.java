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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;

public class TreeUtils {
  private TreeUtils() {
    // empty constructor
  }

  @CheckForNull
  public static Tree firstAncestor(Tree tree, Predicate<Tree> predicate) {
    Tree currentParent = tree.parent();
    while (currentParent != null) {
      if (predicate.test(currentParent)) {
        return currentParent;
      }
      currentParent = currentParent.parent();
    }
    return null;
  }

  @CheckForNull
  public static Tree firstAncestorOfKind(Tree tree, Kind... kinds) {
    return firstAncestor(tree, t -> t.is(kinds));
  }

  public static List<Token> tokens(Tree tree) {
    if (tree.is(Kind.TOKEN)) {
      return Collections.singletonList((Token) tree);
    }
    List<Token> tokens = new ArrayList<>();
    for (Tree child : tree.children()) {
      if (child.is(Kind.TOKEN)) {
        tokens.add(((Token) child));
      } else {
        tokens.addAll(tokens(child));
      }
    }
    return tokens;
  }

  public static boolean hasDescendant(Tree tree, Predicate<Tree> predicate) {
    return tree.children().stream().anyMatch(child -> predicate.test(child) || hasDescendant(child, predicate));
  }
}
