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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.PythonType;

public class ClassTypeBuilder implements TypeBuilder<ClassType> {


  String name;
  Set<Member> members = new HashSet<>();
  List<PythonType> attributes = new ArrayList<>();
  List<PythonType> superClasses = new ArrayList<>();
  List<PythonType> metaClasses = new ArrayList<>();
  LocationInFile definitionLocation;

  @Override
  public ClassType build() {
    return new ClassType(name, members, attributes, superClasses, metaClasses, definitionLocation);
  }

  public ClassTypeBuilder setName(String name) {
    this.name = name;
    return this;
  }

  @Override
  public ClassTypeBuilder withDefinitionLocation(LocationInFile definitionLocation) {
    this.definitionLocation = definitionLocation;
    return this;
  }

  public List<PythonType> superClasses() {
    return superClasses;
  }

  public List<PythonType> metaClasses() {
    return metaClasses;
  }
}
