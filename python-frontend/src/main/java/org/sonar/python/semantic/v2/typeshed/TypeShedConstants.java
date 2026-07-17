/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
  public static final String PYTEST_FIXTURE_REQUEST_FQN = "_pytest.fixtures.FixtureRequest";
  public static final String PYTEST_FIXTURE_REQUEST_GET_FIXTURE_VALUE_FQN = PYTEST_FIXTURE_REQUEST_FQN + ".getfixturevalue";

  private TypeShedConstants() {
  }
}
