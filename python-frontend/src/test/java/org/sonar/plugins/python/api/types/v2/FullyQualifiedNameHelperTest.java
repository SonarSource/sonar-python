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
package org.sonar.plugins.python.api.types.v2;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.types.v2.SpecialFormType;

import static org.assertj.core.api.Assertions.assertThat;

class FullyQualifiedNameHelperTest {
  
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

    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(function1)).contains("foo.bar.func1");
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(nullFunctionType)).isEmpty();
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(class1)).contains("foo.bar.class1");
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(module1)).contains("mod.module1");
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(specialFormType1)).contains("typing.List");
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(unresolvedImport)).contains("imported.module1");
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(unknownType)).isEmpty();
    
    var selfType = SelfType.of(class1);
    assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(selfType)).contains("foo.bar.class1");
  }
}
