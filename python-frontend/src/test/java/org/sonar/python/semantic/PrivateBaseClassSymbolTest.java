/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic;

import java.util.Set;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.TypeShedTypeTable;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.CALLS_REAL_METHODS;
import static org.mockito.Mockito.mockStatic;

class PrivateBaseClassSymbolTest {
  @AfterEach
  void resetVersions() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void inlining_preserves_private_base_identity_and_public_method_names() {
    var base = baseBuilder().addMethods(method("inherited")).build();
    try (var typeshed = mockStatic(TypeShed.class, CALLS_REAL_METHODS)) {
      typeshed.when(() -> TypeShed.classDescriptorWithTypeTableWithFQN("_module.Base"))
        .thenReturn(new TypeShed.ClassDescriptorWithTypeTable(base, TypeShedTypeTable.EMPTY));
      var child = new ClassSymbolImpl(childBuilder().build(), "public");
      assertThat(child.superClassesFqn()).containsExactly("_module.Base");
      assertThat(child.resolveMember("inherited")).get()
        .extracting(Symbol::fullyQualifiedName).isEqualTo("public.Child.inherited");
      assertThat(ClassSymbolImpl.copyFrom("Alias", child).superClassesFqn()).containsExactly("_module.Base");
    }
  }

  @Test
  void unresolved_private_base_does_not_skip_remaining_bases() {
    var base = baseBuilder().addMethods(method("inherited")).build();
    var child = childBuilder().clearSuperClasses().addSuperClasses("_module.Missing").addSuperClasses("_module.Base").build();
    try (var typeshed = mockStatic(TypeShed.class, CALLS_REAL_METHODS)) {
      typeshed.when(() -> TypeShed.classDescriptorWithTypeTableWithFQN("_module.Missing")).thenReturn(null);
      typeshed.when(() -> TypeShed.classDescriptorWithTypeTableWithFQN("_module.Base"))
        .thenReturn(new TypeShed.ClassDescriptorWithTypeTable(base, TypeShedTypeTable.EMPTY));
      assertThat(new ClassSymbolImpl(child, "public").resolveMember("inherited")).isPresent();
    }
  }

  @Test
  void inherited_methods_and_overloads_respect_the_selected_python_version() {
    var overload = SymbolsProtos.OverloadedFunctionSymbol.newBuilder().setName("overloaded").setFullname("_module.Base.overloaded")
      .addValidFor("313")
      .addDefinitions(method("overloaded"))
      .addDefinitions(method("overloaded").addParameters(SymbolsProtos.ParameterSymbol.newBuilder().setName("value")));
    var base = baseBuilder().addMethods(method("stable")).addMethods(method("future").addValidFor("313"))
      .addOverloadedMethods(overload).build();
    try (var typeshed = mockStatic(TypeShed.class, CALLS_REAL_METHODS)) {
      typeshed.when(() -> TypeShed.classDescriptorWithTypeTableWithFQN("_module.Base"))
        .thenReturn(new TypeShed.ClassDescriptorWithTypeTable(base, TypeShedTypeTable.EMPTY));
      ProjectPythonVersion.setCurrentVersions(Set.of(PythonVersionUtils.Version.V_312));
      TypeShed.resetBuiltinSymbols();
      assertThat(new ClassSymbolImpl(childBuilder().build(), "public").declaredMembers())
        .extracting(Symbol::name).containsExactly("stable");
      ProjectPythonVersion.setCurrentVersions(Set.of(PythonVersionUtils.Version.V_313));
      TypeShed.resetBuiltinSymbols();
      assertThat(new ClassSymbolImpl(childBuilder().build(), "public").declaredMembers())
        .extracting(Symbol::name).containsExactlyInAnyOrder("stable", "future", "overloaded");
    }
  }

  private static SymbolsProtos.FunctionSymbol.Builder method(String name) {
    return SymbolsProtos.FunctionSymbol.newBuilder().setName(name);
  }

  private static SymbolsProtos.ClassSymbol.Builder baseBuilder() {
    return SymbolsProtos.ClassSymbol.newBuilder().setName("Base").setFullyQualifiedName("_module.Base").addSuperClasses("object");
  }

  private static SymbolsProtos.ClassSymbol.Builder childBuilder() {
    return SymbolsProtos.ClassSymbol.newBuilder().setName("Child").setFullyQualifiedName("public.Child").addSuperClasses("_module.Base");
  }
}
