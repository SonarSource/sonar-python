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
package org.sonar.python.semantic.v2.typeshed;

public class TypeShedConstants {
  public static final String BUILTINS_FQN = "builtins";
  public static final String BUILTINS_PREFIX = BUILTINS_FQN + ".";
  public static final String BUILTINS_TYPE_FQN = BUILTINS_PREFIX + "type";
  public static final String BUILTINS_NONE_TYPE_FQN = BUILTINS_PREFIX + "NoneType";
  public static final String BUILTINS_TUPLE_FQN = BUILTINS_PREFIX + "tuple";
  public static final String BUILTINS_DICT_FQN = BUILTINS_PREFIX + "dict";

  private TypeShedConstants() {
  }
}
