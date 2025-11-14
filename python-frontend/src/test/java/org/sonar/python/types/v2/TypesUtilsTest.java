/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.types.v2;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;

class TypesUtilsTest {
  @Test
  void testGetFullyQualifiedName() {
    FunctionType function1 = Mockito.mock(FunctionType.class);
    FunctionType nullFunctionType = Mockito.mock(FunctionType.class);
    ClassType class1 = Mockito.mock(ClassType.class);
    UnknownType.UnresolvedImportType unresolvedImport = Mockito.mock(UnknownType.UnresolvedImportType.class);
    ModuleType module1 = Mockito.mock(ModuleType.class);
    SpecialFormType specialFormType1 = Mockito.mock(SpecialFormType.class);
    UnknownType.UnknownTypeImpl unknownType = Mockito.mock(UnknownType.UnknownTypeImpl.class);

    Mockito.when(function1.fullyQualifiedName()).thenReturn("foo.bar.func1");
    Mockito.when(nullFunctionType.fullyQualifiedName()).thenReturn(null);
    Mockito.when(class1.fullyQualifiedName()).thenReturn("foo.bar.class1");
    Mockito.when(module1.fullyQualifiedName()).thenReturn("mod.module1");
    Mockito.when(specialFormType1.fullyQualifiedName()).thenReturn("typing.List");
    Mockito.when(unresolvedImport.importPath()).thenReturn("imported.module1");

    assertThat(TypeUtils.getFullyQualifiedName(function1)).isEqualTo("foo.bar.func1");
    assertThat(TypeUtils.getFullyQualifiedName(nullFunctionType)).isNull();
    assertThat(TypeUtils.getFullyQualifiedName(class1)).isEqualTo("foo.bar.class1");
    assertThat(TypeUtils.getFullyQualifiedName(module1)).isEqualTo("mod.module1");
    assertThat(TypeUtils.getFullyQualifiedName(specialFormType1)).isEqualTo("typing.List");
    assertThat(TypeUtils.getFullyQualifiedName(unresolvedImport)).isEqualTo("imported.module1");
    assertThat(TypeUtils.getFullyQualifiedName(unknownType)).isNull();
  }
}
