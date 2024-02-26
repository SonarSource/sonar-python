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
package org.sonar.plugins.python.api.symbols;

import org.sonar.plugins.python.api.tree.Tree;

public interface Usage {

  default boolean isBindingUsage() {
    return kind() != Kind.OTHER && kind() != Kind.GLOBAL_DECLARATION;
  }

  Tree tree();

  Kind kind();

  enum Kind {
    ASSIGNMENT_LHS,
    COMPOUND_ASSIGNMENT_LHS,
    IMPORT,
    LOOP_DECLARATION,
    COMP_DECLARATION,
    OTHER,
    PARAMETER,
    FUNC_DECLARATION,
    CLASS_DECLARATION,
    EXCEPTION_INSTANCE,
    WITH_INSTANCE,
    GLOBAL_DECLARATION,
    PATTERN_DECLARATION,
    TYPE_PARAM_DECLARATION,
    TYPE_ALIAS_DECLARATION,
  }
}
