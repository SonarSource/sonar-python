/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;

class BasicTypeTableTest {

  private final BasicTypeTable basicTypeTable = new BasicTypeTable(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));

  @Test
  void testGetBuiltinsModule() {
    PythonType builtinsModuleType = basicTypeTable.getBuiltinsModule();
    assertThat(builtinsModuleType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertResolvedMember(builtinsModuleType, "foo", "foo");
  }

  @Test
  void testGetTypeFromTypeFqn() {
    PythonType fooType = basicTypeTable.getType("foo");
    assertThat(fooType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertResolvedMember(fooType, "bar", "foo.bar");
  }

  @Test
  void testGetTypeFromTypeFqnParts() {
    PythonType fooType = basicTypeTable.getType("foo", "bar");
    assertThat(fooType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertResolvedMember(fooType, "qux", "foo.bar.qux");
  }

  @Test
  void testGetTypeFromTypeFqnList() {
    PythonType fooType = basicTypeTable.getType(List.of("foo", "bar"));
    assertThat(fooType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertResolvedMember(fooType, "qux", "foo.bar.qux");
  }

  @Test
  void testGetModuleType() {
    PythonType fooType = basicTypeTable.getModuleType(List.of("foo", "bar"));
    assertThat(fooType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertResolvedMember(fooType, "qux", "foo.bar.qux");
  }

  private static void assertResolvedMember(PythonType builtinsModuleType, String memberName, String resolvedMemberName) {
    Optional<PythonType> resolvedMemberType = builtinsModuleType.resolveMember(memberName);
    assertThat(resolvedMemberType).isPresent();
    assertThat(resolvedMemberType.get()).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertThat(((UnknownType.UnresolvedImportType) resolvedMemberType.get()).importPath()).isEqualTo(resolvedMemberName);
  }
}
