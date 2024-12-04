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
package org.sonar.python.semantic.v2.typeshed;

import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AliasDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.ModuleDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class ModuleSymbolToDescriptorConverter {
  private final ClassSymbolToDescriptorConverter classConverter;
  private final FunctionSymbolToDescriptorConverter functionConverter;
  private final VarSymbolToDescriptorConverter variableConverter;
  private final OverloadedFunctionSymbolToDescriptorConverter overloadedFunctionConverter;
  private final Set<String> projectPythonVersions;

  public ModuleSymbolToDescriptorConverter(Set<PythonVersionUtils.Version> projectPythonVersions) {
    this.projectPythonVersions = projectPythonVersions.stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
    functionConverter = new FunctionSymbolToDescriptorConverter();
    variableConverter = new VarSymbolToDescriptorConverter();
    overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    classConverter = new ClassSymbolToDescriptorConverter(variableConverter, functionConverter, overloadedFunctionConverter, this.projectPythonVersions);
  }

  @CheckForNull
  public ModuleDescriptor convert(@Nullable SymbolsProtos.ModuleSymbol moduleSymbol) {
    if (moduleSymbol == null) {
      return null;
    }

    var name = moduleSymbol.getFullyQualifiedName();
    var fullyQualifiedName = moduleSymbol.getFullyQualifiedName();
    var members = getModuleDescriptors(moduleSymbol);

    return new ModuleDescriptor(name, fullyQualifiedName, members);
  }

  private Map<String, Descriptor> getModuleDescriptors(SymbolsProtos.ModuleSymbol moduleSymbol) {
    var classesStream = moduleSymbol.getClassesList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(classConverter::convert)
      .map(d -> wrapInAliasIfNeeded(d, moduleSymbol.getFullyQualifiedName()))
      .map(Descriptor.class::cast);
    var functionsStream = moduleSymbol.getFunctionsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(functionConverter::convert)
      .map(d -> wrapInAliasIfNeeded(d, moduleSymbol.getFullyQualifiedName()))
      .map(Descriptor.class::cast);
    var overloadedFunctionsStream = moduleSymbol.getOverloadedFunctionsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(overloadedFunctionConverter::convert)
      .map(Descriptor.class::cast);
    var variablesStream = moduleSymbol.getVarsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(variableConverter::convert)
      .map(Descriptor.class::cast);

    return ProtoUtils.disambiguateByName(Stream.of(classesStream, functionsStream, overloadedFunctionsStream, variablesStream));
  }

  private static Descriptor wrapInAliasIfNeeded(Descriptor descriptor, String moduleFullyQualifiedName) {
    String normalizedModuleFqn = moduleFullyQualifiedName;
    if (moduleFullyQualifiedName.startsWith("builtins")) {
      normalizedModuleFqn = moduleFullyQualifiedName.substring("builtins".length());
    }
    String descriptorFqn = descriptor.fullyQualifiedName();
    if (descriptorFqn == null) {
      return descriptor;
    }
    if (!descriptorFqn.startsWith(normalizedModuleFqn)) {
      String aliasFqn = normalizedModuleFqn + "." + descriptor.name();
      return new AliasDescriptor(descriptor.name(), aliasFqn, descriptor);
    }
    return descriptor;
  }

}
