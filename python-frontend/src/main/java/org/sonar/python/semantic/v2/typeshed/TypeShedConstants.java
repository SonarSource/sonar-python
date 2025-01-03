/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
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
