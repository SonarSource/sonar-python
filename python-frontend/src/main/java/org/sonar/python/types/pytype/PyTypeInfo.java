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
package org.sonar.python.types.pytype;

import com.google.gson.annotations.SerializedName;
import java.util.Optional;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.PyTypeDetailedInfo;

public record PyTypeInfo(
  String text,
  @SerializedName("start_line")
  int startLine,
  @SerializedName("start_col")
  int startCol,
  @SerializedName("syntax_role")
  String syntaxRole,
  PyTypeDetailedInfo type,
  @SerializedName("short_type")
  String shortType,
  @SerializedName("type_details")
  BaseType baseType) {

  public InferredType inferredType() {
    return Optional.of(this)
      .map(PyTypeInfo::type)
      .map(PyTypeDetailedInfo::inferredType)
      .orElseGet(InferredTypes::anyType);
  }
}
