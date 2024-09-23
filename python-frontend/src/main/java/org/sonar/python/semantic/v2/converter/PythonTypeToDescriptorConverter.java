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
package org.sonar.python.semantic.v2.converter;

import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.UnknownDescriptor;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnknownType;

public class PythonTypeToDescriptorConverter {

  private PythonTypeToDescriptorConverter() {
    // Utility class
  }

  public static Descriptor convert(PythonType pythonType, SymbolV2 symbol) {
    String fqn = "dummyFqn";
    if (pythonType instanceof UnknownType) {
      return new UnknownDescriptor(pythonType.name(), fqn);
    }
//    if (pythonType instanceof ObjectType objectType) {
//      return new VariableDescriptor(symbol.name(), fqn, fqn); // Replace 2nd fqn with the fqn of the objecttype.type
//    }
    if (pythonType  instanceof FunctionType functionType) {
      var builder = new FunctionDescriptor.FunctionDescriptorBuilder()
        .withName(functionType.name())
        .withFullyQualifiedName(functionType.fullyQualifiedName())
        .withIsAsynchronous(functionType.isAsynchronous())
        .withIsInstanceMethod(functionType.isInstanceMethod())
        // TODO represent decorators SONARPY-1772
        .withHasDecorators(functionType.hasDecorators())
        .withParameters(functionType.parameters().stream().map(PythonTypeToDescriptorConverter::convertParameter).toList())
        .withDefinitionLocation(functionType.definitionLocation().orElse(null))
        .withAnnotatedReturnTypeName(functionType.returnType().fullyQualifiedName());
      return builder.build();
    }

    return null;
  }
  
  private static FunctionDescriptor.Parameter convertParameter(ParameterV2 parameter) {
    var parameterType = parameter.declaredType().type();
    var annotatedType = parameterType instanceof ObjectType objectType ? objectType.unwrappedType().fullyQualifiedName() : parameterType.fullyQualifiedName();
    return new FunctionDescriptor.Parameter(parameter.name(),
      annotatedType,
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isPositionalVariadic(), 
      parameter.isKeywordVariadic(),
      parameter.location()
    );
  }
}
