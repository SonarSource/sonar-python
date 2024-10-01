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

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;

class SymbolsModuleTypeProviderTest {

  SymbolsModuleTypeProvider symbolsModuleTypeProvider;

  @BeforeEach
  void setUp() {
    ProjectLevelSymbolTable empty = ProjectLevelSymbolTable.empty();
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(empty);
    LazyTypesContext lazyTypesContext = projectLevelTypeTable.lazyTypesContext();
    symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(empty, lazyTypesContext);
  }

  @Test
  void getSymbolTypeFqnSpecialForm() {
    var type = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.INSTANCE).setFullyQualifiedName("typing._SpecialForm").build();
    assertThat(symbolsModuleTypeProvider.getSymbolTypeFqn(type)).isEmpty();
  }

  @Test
  void getSymbolTypeFqnTypedDict() {
    var type = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.TYPED_DICT).build();
    assertThat(symbolsModuleTypeProvider.getSymbolTypeFqn(type)).containsExactly("dict");
  }

  @Test
  void getSymbolTypeFqnType() {
    var type = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.TYPE).build();
    assertThat(symbolsModuleTypeProvider.getSymbolTypeFqn(type)).containsExactly("type");
  }

  @Test
  void getSymbolTypeFqnTypeAlias() {
    var type = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.INSTANCE).setFullyQualifiedName("builtins.float").build();
    var typeAlias = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.TYPE_ALIAS).addArgs(type).build();
    assertThat(symbolsModuleTypeProvider.getSymbolTypeFqn(typeAlias)).containsExactly("float");
  }

  @Test
  void getSymbolTypeObject() {
    var type = SymbolsProtos.Type.newBuilder().setKind(SymbolsProtos.TypeKind.INSTANCE).setFullyQualifiedName("object").build();
    assertThat(symbolsModuleTypeProvider.getSymbolTypeFqn(type)).isEmpty();
  }
}
