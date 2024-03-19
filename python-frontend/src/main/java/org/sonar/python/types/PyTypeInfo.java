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
package org.sonar.python.types;

import com.google.gson.annotations.SerializedName;
import java.util.Optional;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.pytype.BaseType;

public class PyTypeInfo {
  private final String text;
  @SerializedName("start_line")
  private final int startLine;
  @SerializedName("start_col")
  private final int startCol;
  @SerializedName("syntax_role")
  private final String syntaxRole;
  private final PyTypeDetailedInfo type;
  @SerializedName("short_type")
  private final String shortType;
  @SerializedName("type_details")
  private final BaseType baseType;

  public PyTypeInfo(String text, int startLine, int startCol, String syntaxRole, PyTypeDetailedInfo type, String shortType, BaseType baseType) {
    this.text = text;
    this.startLine = startLine;
    this.startCol = startCol;
    this.syntaxRole = syntaxRole;
    this.type = type;
    this.shortType = shortType;
    this.baseType = baseType;
  }

  public String text() {
    return text;
  }

  public int startLine() {
    return startLine;
  }

  public int startCol() {
    return startCol;
  }

  public String syntaxRole() {
    return syntaxRole;
  }

  public PyTypeDetailedInfo type() {
    return type;
  }

  public String shortType() {
    return shortType;
  }

  public BaseType baseType() {
    return baseType;
  }

  public InferredType inferredType() {
    return Optional.of(this)
      .map(PyTypeInfo::type)
      .map(PyTypeDetailedInfo::inferredType)
      .orElseGet(InferredTypes::anyType);
  }
}
