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
import org.sonar.python.types.TypeShedTypeTable;

public class ModuleSymbolToDescriptorConverter {
  private final Set<String> projectSemanticPythonVersions;

  public ModuleSymbolToDescriptorConverter(Set<PythonVersionUtils.SemanticVersion> projectSemanticPythonVersions) {
    this.projectSemanticPythonVersions = projectSemanticPythonVersions.stream()
      .map(PythonVersionUtils.SemanticVersion::serializedValue)
      .collect(Collectors.toSet());
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
    var typeTable = TypeShedTypeTable.from(moduleSymbol);
    var functionConverter = new FunctionSymbolToDescriptorConverter(typeTable);
    var variableConverter = new VarSymbolToDescriptorConverter(typeTable);
    var overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    var classConverter = new ClassSymbolToDescriptorConverter(
      variableConverter, functionConverter, overloadedFunctionConverter, projectSemanticPythonVersions);
    var classesStream = moduleSymbol.getClassesList()
      .stream()
      .filter(d -> ProtoUtils.isValidForSemanticPythonVersion(d.getValidForList(), projectSemanticPythonVersions))
      .map(s -> classConverter.convert(s, moduleSymbol.getFullyQualifiedName()))
      .map(d -> wrapInAliasIfNeeded(d, moduleSymbol.getFullyQualifiedName()))
      .map(Descriptor.class::cast);
    var functionsStream = moduleSymbol.getFunctionsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForSemanticPythonVersion(d.getValidForList(), projectSemanticPythonVersions))
      .map(s -> functionConverter.convert(s, false, moduleSymbol.getFullyQualifiedName()))
      .map(d -> wrapInAliasIfNeeded(d, moduleSymbol.getFullyQualifiedName()))
      .map(Descriptor.class::cast);
    var overloadedFunctionsStream = moduleSymbol.getOverloadedFunctionsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForSemanticPythonVersion(d.getValidForList(), projectSemanticPythonVersions))
      .map(s -> overloadedFunctionConverter.convert(s, false, moduleSymbol.getFullyQualifiedName()))
      .map(Descriptor.class::cast);
    var variablesStream = moduleSymbol.getVarsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForSemanticPythonVersion(d.getValidForList(), projectSemanticPythonVersions))
      .map(s -> variableConverter.convert(s, moduleSymbol.getFullyQualifiedName()))
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
