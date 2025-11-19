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
package org.sonar.python.semantic.v2;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.v2.typetable.CachedTypeTable;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CachedTypeTableTest {

  @Test
  void allGetTypeVariants_shouldShareSameCache() {
    TypeTable delegate = mock(TypeTable.class);
    PythonType mockType = mock(PythonType.class);
    when(delegate.getType(List.of("foo", "bar"))).thenReturn(mockType);

    CachedTypeTable cachedTypeTable = new CachedTypeTable(delegate);

    PythonType firstCall = cachedTypeTable.getType("foo.bar");
    PythonType secondCall = cachedTypeTable.getType("foo", "bar");
    PythonType thirdCall = cachedTypeTable.getType(List.of("foo", "bar"));

    assertThat(firstCall).isSameAs(mockType);
    assertThat(secondCall).isSameAs(mockType);
    assertThat(thirdCall).isSameAs(mockType);
    verify(delegate, times(1)).getType(List.of("foo", "bar"));
  }

  @Test
  void getModuleType_shouldCacheResults() {
    TypeTable delegate = mock(TypeTable.class);
    PythonType mockModuleType = mock(PythonType.class);
    List<String> parts = List.of("foo", "bar");
    when(delegate.getModuleType(parts)).thenReturn(mockModuleType);

    CachedTypeTable cachedTypeTable = new CachedTypeTable(delegate);

    PythonType firstCall = cachedTypeTable.getModuleType(parts);
    PythonType secondCall = cachedTypeTable.getModuleType(parts);

    assertThat(firstCall).isSameAs(mockModuleType);
    assertThat(secondCall).isSameAs(mockModuleType);
    verify(delegate, times(1)).getModuleType(parts);
  }

  @Test
  void getModuleType_shouldUseSeparateCacheFromGetType() {
    TypeTable delegate = mock(TypeTable.class);
    PythonType mockType = mock(PythonType.class);
    PythonType mockModuleType = mock(PythonType.class);
    List<String> parts = List.of("foo", "bar");
    when(delegate.getType(parts)).thenReturn(mockType);
    when(delegate.getModuleType(parts)).thenReturn(mockModuleType);

    CachedTypeTable cachedTypeTable = new CachedTypeTable(delegate);

    PythonType typeCall = cachedTypeTable.getType(parts);
    PythonType moduleTypeCall = cachedTypeTable.getModuleType(parts);

    assertThat(typeCall).isSameAs(mockType);
    assertThat(moduleTypeCall).isSameAs(mockModuleType);
    verify(delegate, times(1)).getType(parts);
    verify(delegate, times(1)).getModuleType(parts);
  }

  @Test
  void getBuiltinsModule_shouldNotBeCached() {
    TypeTable delegate = mock(TypeTable.class);
    PythonType mockBuiltinsModule = mock(PythonType.class);
    when(delegate.getBuiltinsModule()).thenReturn(mockBuiltinsModule);

    CachedTypeTable cachedTypeTable = new CachedTypeTable(delegate);

    PythonType firstCall = cachedTypeTable.getBuiltinsModule();
    PythonType secondCall = cachedTypeTable.getBuiltinsModule();
    PythonType thirdCall = cachedTypeTable.getBuiltinsModule();

    assertThat(firstCall).isSameAs(mockBuiltinsModule);
    assertThat(secondCall).isSameAs(mockBuiltinsModule);
    assertThat(thirdCall).isSameAs(mockBuiltinsModule);
    verify(delegate, times(3)).getBuiltinsModule();
  }
}
