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
  public static final String EXCEPTION = "Exception";
  public static final String BASE_EXCEPTION = "BaseException";

  public static final String OBJECT_TYPE = "object";

  private BuiltinTypes() {
  }

}
