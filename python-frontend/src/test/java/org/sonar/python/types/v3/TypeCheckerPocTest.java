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
package org.sonar.python.types.v3;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.ObjectTypeBuilder;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.types.v2.TypeUtils;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v3.TypeCheckerPoc.UnspecializedTypeCheckerBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v3.TypeCheckerPocPredicates.isObject;

class TypeCheckerPocTest {

  @Test
  void simpleClassType() {
    var projectLeveLTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var builderContext = new TypeCheckerPoc.TypeCheckerBuilderContext(projectLeveLTypeTable);
    var typeChecker = new UnspecializedTypeCheckerBuilder(builderContext)
      .with(TypeCheckerPocPredicates.isClass())
      .build();

    var classType = new ClassTypeBuilder().withName("MyClass").build();
    var objectType = new ObjectTypeBuilder().build();

    assertThat(typeChecker.isTrue(classType)).isTrue();
    assertThat(typeChecker.isTrue(objectType)).isFalse();
  }

  @Test
  void simpleIsObject() {
    var projectLeveLTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var builderContext = new TypeCheckerPoc.TypeCheckerBuilderContext(projectLeveLTypeTable);
    var typeChecker = new UnspecializedTypeCheckerBuilder(builderContext)
      .with(isObject("builtins.float"))
      .build();

    var floatType = TypeUtils.ensureWrappedObjectType(projectLeveLTypeTable.getType("builtins.float"));
    var classType = new ClassTypeBuilder().withName("MyClass").build();

    assertThat(typeChecker.isTrue(floatType)).isTrue();
    assertThat(typeChecker.isTrue(classType)).isFalse();
  }

  @Test
  void simpleAnyCandidate() {
    var projectLeveLTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var builderContext = new TypeCheckerPoc.TypeCheckerBuilderContext(projectLeveLTypeTable);
    var typeCheckerAny = new UnspecializedTypeCheckerBuilder(builderContext)
      .with(isObject("builtins.float").anyCandidate())
      .build();
    var typeCheckerDefault = new UnspecializedTypeCheckerBuilder(builderContext)
      .with(isObject("builtins.float"))
      .build();

    var objType1 = TypeUtils.ensureWrappedObjectType(projectLeveLTypeTable.getType("NoneType"));
    var objFloat = TypeUtils.ensureWrappedObjectType(projectLeveLTypeTable.getType("builtins.float"));
    var objType2 = TypeUtils.ensureWrappedObjectType(projectLeveLTypeTable.getType("builtins.int"));

    var unionType = UnionType.or(List.of(objType1, objFloat, objType2));

    assertThat(typeCheckerAny.isTrue(unionType)).isTrue();
    assertThat(typeCheckerDefault.isTrue(unionType)).isFalse();
  }

}
