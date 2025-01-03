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
package org.sonar.plugins.python.api.types;

public class BuiltinTypes {

  public static final String NONE_TYPE = "NoneType";
  public static final String STR = "str";
  public static final String BOOL = "bool";
  public static final String INT = "int";
  public static final String FLOAT = "float";
  public static final String COMPLEX = "complex";
  public static final String TUPLE = "tuple";
  public static final String LIST = "list";
  public static final String SET = "set";
  public static final String DICT = "dict";

  // https://docs.python.org/3/library/stdtypes.html#bytes-objects
  public static final String BYTES = "bytes";

  public static final String EXCEPTION = "Exception";
  public static final String BASE_EXCEPTION = "BaseException";

  public static final String OBJECT_TYPE = "object";

  private BuiltinTypes() {
  }

}
