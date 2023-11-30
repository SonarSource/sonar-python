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
package org.sonar.python.checks.utils;

import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

public class StringLiteralValuesCollector extends BaseTreeVisitor {
  private final Set<String> stringLiteralValues = new HashSet<>();

  public void collect(Tree tree) {
    stringLiteralValues.clear();
    tree.accept(this);
  }

  @Override
  public void visitStringLiteral(StringLiteral pyStringLiteralTree) {
    stringLiteralValues.add(pyStringLiteralTree.trimmedQuotesValue());
  }

  public boolean anyMatches(Predicate<String> predicate) {
    return stringLiteralValues.stream().anyMatch(predicate);
  }
}
