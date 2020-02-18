/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.types;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.types.InferredType;

public class InferredTypes {

  public static final InferredType INT = runtimeType("int");
  public static final InferredType FLOAT = runtimeType("float");
  public static final InferredType COMPLEX = runtimeType("complex");

  public static final InferredType STR = runtimeType("str");
  public static final InferredType BYTES = runtimeType("bytes");

  public static final InferredType SET = runtimeType("set");
  public static final InferredType DICT = runtimeType("dict");
  public static final InferredType LIST = runtimeType("list");
  public static final InferredType TUPLE = runtimeType("tuple");
  public static final InferredType GENERATOR = runtimeType("generator");

  private InferredTypes() {
  }

  public static InferredType anyType() {
    return AnyType.ANY;
  }

  public static InferredType runtimeType(@Nullable String fullyQualifiedName) {
    if (fullyQualifiedName == null) {
      return anyType();
    }
    return new RuntimeType(fullyQualifiedName);
  }

}
