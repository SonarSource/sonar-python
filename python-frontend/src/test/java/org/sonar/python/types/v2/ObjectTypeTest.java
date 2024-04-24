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
package org.sonar.python.types.v2;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;

import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;

import static org.assertj.core.api.Assertions.assertThat;


public class ObjectTypeTest {

  @Test
  void simpleObject() {
    FileInput fileInput = parseAndInferTypes("""
      class A: ...
      a = A()
      a
      """
    );
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    ObjectType objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();

    assertThat(objectType.displayName()).isEqualTo("A");
    assertThat(objectType.isCompatibleWith(classType)).isTrue();
    assertThat(objectType.hasMember("foo")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void simpleObjectWithMember() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a
      """
    );
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    ObjectType objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();

    assertThat(objectType.displayName()).isEqualTo("A");
    assertThat(objectType.isCompatibleWith(classType)).isTrue();
    assertThat(objectType.hasMember("foo")).isEqualTo(TriBool.TRUE);
  }

  @Test
  void reassignedObject() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a = B()
      a
      """
    );
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    PythonType aType = ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();

    assertThat(aType).isEqualTo(PythonType.UNKNOWN);
    assertThat(aType.isCompatibleWith(classType)).isTrue();
    assertThat(aType.hasMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void objectType_of_unknown() {
    // TODO: Ensure this is the behavior we want (do we even want it possible to have object of unknown? Maybe replace with UnionType when implemented
    ObjectType objectType = new ObjectType(PythonType.UNKNOWN, List.of(), List.of());
    assertThat(objectType.hasMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }


  public static FileInput parseAndInferTypes(String... code) {
    FileInput fileInput = parseWithoutSymbols(code);
    fileInput.accept(new SymbolTableBuilderV2());
    fileInput.accept(new TypeInferenceV2(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));
    return fileInput;
  }
}
