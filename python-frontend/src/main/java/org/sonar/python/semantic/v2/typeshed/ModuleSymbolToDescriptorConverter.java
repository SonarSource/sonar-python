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
package org.sonar.python.semantic.v2.typeshed;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.ModuleDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class ModuleSymbolToDescriptorConverter {
  private final ClassSymbolToDescriptorConverter classConverter;
  private final FunctionSymbolToDescriptorConverter functionConverter;
  private final VarSymbolToDescriptorConverter variableConverter;
  private final OverloadedFunctionSymbolToDescriptorConverter overloadedFunctionConverter;
  private final Set<String> supportedPythonVersions;

  public ModuleSymbolToDescriptorConverter() {
    functionConverter = new FunctionSymbolToDescriptorConverter();
    variableConverter = new VarSymbolToDescriptorConverter();
    overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    classConverter = new ClassSymbolToDescriptorConverter(variableConverter, functionConverter, overloadedFunctionConverter);
    supportedPythonVersions = ProjectPythonVersion.currentVersionValues();
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
    // TODO: Use a common proxy interface Descriptor instead of using Object
    Map<String, Set<Descriptor>> protoSymbolsByName = new HashMap<>();
    moduleSymbol.getClassesList()
      .stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .map(classConverter::convert)
      .forEach(descriptor -> protoSymbolsByName.computeIfAbsent(descriptor.name(), d -> new HashSet<>()).add(descriptor));
    moduleSymbol.getFunctionsList()
      .stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .map(functionConverter::convert)
      .forEach(descriptor -> protoSymbolsByName.computeIfAbsent(descriptor.name(), d -> new HashSet<>()).add(descriptor));
    moduleSymbol.getOverloadedFunctionsList()
      .stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .map(overloadedFunctionConverter::convert)
      .forEach(descriptor -> protoSymbolsByName.computeIfAbsent(descriptor.name(), d -> new HashSet<>()).add(descriptor));
   moduleSymbol.getVarsList()
      .stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .map(variableConverter::convert)
      .forEach(descriptor -> protoSymbolsByName.computeIfAbsent(descriptor.name(), d -> new HashSet<>()).add(descriptor));

    var descriptorsByName = new HashMap<String, Descriptor>();

    protoSymbolsByName.forEach((name, descriptors) -> {
      Descriptor disambiguatedDescriptor = disambiguateSymbolsWithSameName(descriptors);
      descriptorsByName.put(name, disambiguatedDescriptor);
    });

    return descriptorsByName;
  }

  private Descriptor disambiguateSymbolsWithSameName(Set<Descriptor> descriptors) {
    if (descriptors.size() > 1) {
      return AmbiguousDescriptor.create(descriptors);
    }
    return descriptors.iterator().next();
  }

  private boolean isValidForProjectPythonVersion(List<String> validForPythonVersions) {
    if (validForPythonVersions.isEmpty()) {
      return true;
    }
    // TODO: SONARPY-1522 - remove this workaround when we will have all the stubs for Python 3.12.
    if (supportedPythonVersions.stream().allMatch(PythonVersionUtils.Version.V_312.serializedValue()::equals)
        && validForPythonVersions.contains(PythonVersionUtils.Version.V_311.serializedValue())) {
      return true;
    }
    HashSet<String> intersection = new HashSet<>(validForPythonVersions);
    intersection.retainAll(supportedPythonVersions);
    return !intersection.isEmpty();
  }

}
