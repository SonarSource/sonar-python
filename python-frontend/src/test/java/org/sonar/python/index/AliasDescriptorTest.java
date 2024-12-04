/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.index;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class AliasDescriptorTest {

  @Test
  void aliasDescriptorOfClass() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    Map<String, Descriptor> stringDescriptorMap = projectLevelSymbolTable.typeShedDescriptorsProvider().descriptorsForModule("fastapi.responses");
    AliasDescriptor aliasDescriptor = (AliasDescriptor) stringDescriptorMap.get("Response");
    assertThat(aliasDescriptor.name()).isEqualTo("Response");
    assertThat(aliasDescriptor.fullyQualifiedName()).isEqualTo("fastapi.responses.Response");
    assertThat(aliasDescriptor.kind()).isEqualTo(Descriptor.Kind.ALIAS);

    Descriptor originalDescriptor = aliasDescriptor.originalDescriptor();
    assertThat(originalDescriptor.name()).isEqualTo("Response");
    assertThat(originalDescriptor.fullyQualifiedName()).isEqualTo("starlette.responses.Response");
    assertThat(originalDescriptor.kind()).isEqualTo(Descriptor.Kind.CLASS);

    Symbol convertedSymbol = DescriptorUtils.symbolFromDescriptor(aliasDescriptor, projectLevelSymbolTable, null, new HashMap<>(), new HashMap<>());
    assertThat(convertedSymbol)
      .isInstanceOf(ClassSymbol.class)
      .hasFieldOrPropertyWithValue("name", "Response")
      .hasFieldOrPropertyWithValue("fullyQualifiedName", "fastapi.responses.Response");
  }

  @Test
  void aliasDescriptorOfFunction() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    Map<String, Descriptor> stringDescriptorMap = projectLevelSymbolTable.typeShedDescriptorsProvider().descriptorsForModule("fastapi.concurrency");
    AliasDescriptor aliasDescriptor = (AliasDescriptor) stringDescriptorMap.get("run_in_threadpool");
    assertThat(aliasDescriptor.name()).isEqualTo("run_in_threadpool");
    assertThat(aliasDescriptor.fullyQualifiedName()).isEqualTo("fastapi.concurrency.run_in_threadpool");
    assertThat(aliasDescriptor.kind()).isEqualTo(Descriptor.Kind.ALIAS);

    Descriptor originalDescriptor = aliasDescriptor.originalDescriptor();
    assertThat(originalDescriptor.name()).isEqualTo("run_in_threadpool");
    assertThat(originalDescriptor.fullyQualifiedName()).isEqualTo("starlette.concurrency.run_in_threadpool");
    assertThat(originalDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);

    Symbol convertedSymbol = DescriptorUtils.symbolFromDescriptor(aliasDescriptor, projectLevelSymbolTable, null, new HashMap<>(), new HashMap<>());
    assertThat(convertedSymbol)
      .isInstanceOf(FunctionSymbol.class)
      .hasFieldOrPropertyWithValue("name", "run_in_threadpool")
      .hasFieldOrPropertyWithValue("fullyQualifiedName", "fastapi.concurrency.run_in_threadpool");
  }

  @Test
  void aliasDescriptorOfVariableIsNotSupported() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    AliasDescriptor aliasDescriptor = new AliasDescriptor("alias", "original", new VariableDescriptor("original", "original", null));
    assertThatThrownBy(() -> DescriptorUtils.symbolFromDescriptor(aliasDescriptor, projectLevelSymbolTable, null, new HashMap<>(), new HashMap<>()))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Error while recreating a descriptor from an alias: Unexpected alias kind: VARIABLE");
  }
}
