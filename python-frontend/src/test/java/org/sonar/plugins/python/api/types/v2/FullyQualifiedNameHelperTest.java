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
package org.sonar.plugins.python.api.types.v2;

import org.junit.jupiter.api.Test;
import org.sonar.python.types.v2.SpecialFormType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class FullyQualifiedNameHelperTest {
  
  @Test
  void testGetFullyQualifiedName() {
    FunctionType function1 = mock(FunctionType.class);
    FunctionType nullFunctionType = mock(FunctionType.class);
    ClassType class1 = mock(ClassType.class);
    UnknownType.UnresolvedImportType unresolvedImport = mock(UnknownType.UnresolvedImportType.class);
    ModuleType module1 = mock(ModuleType.class);
    SpecialFormType specialFormType1 = mock(SpecialFormType.class);
    UnknownType.UnknownTypeImpl unknownType = mock(UnknownType.UnknownTypeImpl.class);

    when(function1.fullyQualifiedName()).thenReturn("foo.bar.func1");
    when(nullFunctionType.fullyQualifiedName()).thenReturn(null);
    when(class1.fullyQualifiedName()).thenReturn("foo.bar.class1");
    when(module1.fullyQualifiedName()).thenReturn("mod.module1");
    when(specialFormType1.fullyQualifiedName()).thenReturn("typing.List");
    when(unresolvedImport.importPath()).thenReturn("imported.module1");

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
