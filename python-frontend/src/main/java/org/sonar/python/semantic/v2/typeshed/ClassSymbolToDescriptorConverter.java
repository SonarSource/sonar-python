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

import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class ClassSymbolToDescriptorConverter {

  private final VarSymbolToDescriptorConverter varConverter;
  private final FunctionSymbolToDescriptorConverter functionConverter;
  private final OverloadedFunctionSymbolToDescriptorConverter overloadedFunctionConverter;
  private final Set<String> projectPythonVersions;

  public ClassSymbolToDescriptorConverter(VarSymbolToDescriptorConverter varConverter,
    FunctionSymbolToDescriptorConverter functionConverter,
    OverloadedFunctionSymbolToDescriptorConverter overloadedFunctionConverter, Set<String> projectPythonVersions) {
    this.varConverter = varConverter;
    this.functionConverter = functionConverter;
    this.overloadedFunctionConverter = overloadedFunctionConverter;
    this.projectPythonVersions = projectPythonVersions;
  }

  public ClassDescriptor convert(SymbolsProtos.ClassSymbol classSymbol) {
    var fullyQualifiedName = TypeShedUtils.normalizedFqn(classSymbol.getFullyQualifiedName());
    var superClasses = classSymbol.getSuperClassesList().stream().map(TypeShedUtils::normalizedFqn).toList();
    var metaclassName = TypeShedUtils.normalizedFqn(classSymbol.getMetaclassName());

    var variableDescriptors = classSymbol.getAttributesList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(varConverter::convert);

    var functionDescriptors = classSymbol.getMethodsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(s -> functionConverter.convert(s, true));

    var overloadedFunctionDescriptors = classSymbol.getOverloadedMethodsList()
      .stream()
      .filter(d -> ProtoUtils.isValidForPythonVersion(d.getValidForList(), projectPythonVersions))
      .map(s -> overloadedFunctionConverter.convert(s, true));

    var members = ProtoUtils.disambiguateByName(Stream.of(variableDescriptors, functionDescriptors, overloadedFunctionDescriptors))
      .values().stream().map(Descriptor.class::cast).collect(Collectors.toSet());

    return new ClassDescriptor.ClassDescriptorBuilder()
      .withName(classSymbol.getName())
      .withFullyQualifiedName(fullyQualifiedName)
      .withSuperClasses(superClasses)
      .withMetaclassFQN(metaclassName)
      .withHasMetaClass(classSymbol.getHasMetaclass())
      .withHasDecorators(classSymbol.getHasDecorators())
      .withSupportsGenerics(classSymbol.getIsGeneric())
      .withMembers(members)
      .build();
  }

}
