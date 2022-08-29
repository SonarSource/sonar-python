/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.tests;

import java.util.Set;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.python.tree.TreeUtils;

public class PytestUtils {

  private PytestUtils() {

  }

  private static final String LIB_NAME = "pytest";
  public static final Set<String> RAISE_METHODS = Set.of("raises");

  public static boolean isPytest(QualifiedExpression qualifiedExpression) {
    return TreeUtils.getSymbolFromTree(qualifiedExpression.qualifier())
      .stream()
      .map(Symbol::name)
      .anyMatch(name -> name.equals(LIB_NAME));
  }
}
