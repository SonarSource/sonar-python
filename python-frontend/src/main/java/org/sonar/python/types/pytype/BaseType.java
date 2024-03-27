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

import org.sonar.python.types.pytype.json.JsonSubtype;
import org.sonar.python.types.pytype.json.JsonType;

@JsonType(
  property = "$class",
  subtypes = {
    @JsonSubtype(name = "Primitive", child = PrimitiveType.class),
    @JsonSubtype(name = "ClassType", child = ClassType.class),
    @JsonSubtype(name = "Module", child = Module.class),
    @JsonSubtype(name = "Alias", child = Alias.class),
    @JsonSubtype(name = "GenericType", child = GenericType.class),
    @JsonSubtype(name = "CallableType", child = CallableType.class),
    @JsonSubtype(name = "TupleType", child = TupleType.class),
    @JsonSubtype(name = "AnythingType", child = AnythingType.class),
    @JsonSubtype(name = "NothingType", child = NothingType.class),
    @JsonSubtype(name = "UnionType", child = UnionType.class),
    @JsonSubtype(name = "TypeParameter", child = TypeParameter.class),
  }
)
public interface BaseType {
}
