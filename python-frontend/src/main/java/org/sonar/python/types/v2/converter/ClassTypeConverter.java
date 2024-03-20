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
package org.sonar.python.types.v2.converter;

import java.util.List;
import java.util.Map;
import org.sonar.python.types.pytype.ClassType;
import org.sonar.python.types.v2.ExactType;
import org.sonar.python.types.v2.PythonType;

public class ClassTypeConverter implements PyTypeConverter<ClassType, ExactType> {
  @Override
  public ExactType convert(ClassType from) {
    return new ExactType(from.name(), List.of(), Map.of());
  }
}
