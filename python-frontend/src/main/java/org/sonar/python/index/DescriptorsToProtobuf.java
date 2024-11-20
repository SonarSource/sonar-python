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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class DescriptorsToProtobuf {

  private DescriptorsToProtobuf() {}

  public static DescriptorsProtos.ModuleDescriptor toProtobufModuleDescriptor(Set<Descriptor> descriptors) {
    List<DescriptorsProtos.ClassDescriptor> classDescriptors = new ArrayList<>();
    List<DescriptorsProtos.FunctionDescriptor> functionDescriptors = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> varDescriptors = new ArrayList<>();
    List<DescriptorsProtos.AmbiguousDescriptor> ambiguousDescriptors = new ArrayList<>();
    for (Descriptor descriptor : descriptors) {
      Descriptor.Kind kind = descriptor.kind();
      if (kind == Descriptor.Kind.CLASS) {
        classDescriptors.add(toProtobuf(((ClassDescriptor) descriptor)));
      } else if (kind == Descriptor.Kind.FUNCTION) {
        functionDescriptors.add(toProtobuf((FunctionDescriptor) descriptor));
      } else if (kind == Descriptor.Kind.VARIABLE) {
        varDescriptors.add(toProtobuf((VariableDescriptor) descriptor));
      } else {
        ambiguousDescriptors.add(toProtobuf((AmbiguousDescriptor) descriptor));
      }
    }
    return DescriptorsProtos.ModuleDescriptor.newBuilder()
      .addAllClassDescriptors(classDescriptors)
      .addAllFunctionDescriptors(functionDescriptors)
      .addAllVarDescriptors(varDescriptors)
      .addAllAmbiguousDescriptors(ambiguousDescriptors)
      .build();
  }

  public static DescriptorsProtos.AmbiguousDescriptor toProtobuf(AmbiguousDescriptor ambiguousDescriptor) {
    List<DescriptorsProtos.FunctionDescriptor> functionDescriptors = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> variableDescriptors = new ArrayList<>();
    List<DescriptorsProtos.ClassDescriptor> classDescriptors = new ArrayList<>();
    for (Descriptor descriptor : ambiguousDescriptor.alternatives()) {
      Descriptor.Kind kind = descriptor.kind();
      if (kind == Descriptor.Kind.FUNCTION) {
        functionDescriptors.add(toProtobuf((FunctionDescriptor) descriptor));
      } else if (kind == Descriptor.Kind.VARIABLE) {
        variableDescriptors.add(toProtobuf((VariableDescriptor) descriptor));
      } else {
        classDescriptors.add(toProtobuf((ClassDescriptor) descriptor));
      }
    }
    DescriptorsProtos.AmbiguousDescriptor.Builder builder = DescriptorsProtos.AmbiguousDescriptor.newBuilder();
    builder.setName(ambiguousDescriptor.name())
      .addAllClassDescriptors(classDescriptors)
      .addAllFunctionDescriptors(functionDescriptors)
      .addAllVarDescriptors(variableDescriptors);
    String fullyQualifiedName = ambiguousDescriptor.fullyQualifiedName();
    if (fullyQualifiedName != null) {
      builder.setFullyQualifiedName(fullyQualifiedName);
    }
    return builder.build();
  }

  public static DescriptorsProtos.ClassDescriptor toProtobuf(ClassDescriptor classDescriptor) {
    List<DescriptorsProtos.FunctionDescriptor> functionMembers = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> variableMembers = new ArrayList<>();
    List<DescriptorsProtos.AmbiguousDescriptor> ambiguousMembers = new ArrayList<>();
    List<DescriptorsProtos.ClassDescriptor> classMembers = new ArrayList<>();
    for (Descriptor member : classDescriptor.members()) {
      Descriptor.Kind kind = member.kind();
      if (kind == Descriptor.Kind.FUNCTION) {
        functionMembers.add(toProtobuf(((FunctionDescriptor) member)));
      } else if (kind == Descriptor.Kind.VARIABLE) {
        variableMembers.add(toProtobuf(((VariableDescriptor) member)));
      } else if (kind == Descriptor.Kind.AMBIGUOUS) {
        ambiguousMembers.add(toProtobuf((AmbiguousDescriptor) member));
      } else {
        classMembers.add(toProtobuf((ClassDescriptor) member));
      }
    }
    DescriptorsProtos.ClassDescriptor.Builder builder = DescriptorsProtos.ClassDescriptor.newBuilder()
      .setName(classDescriptor.name())
      .setFullyQualifiedName(classDescriptor.fullyQualifiedName())
      .addAllSuperClasses(classDescriptor.superClasses())
      .addAllFunctionMembers(functionMembers)
      .addAllVarMembers(variableMembers)
      .addAllAmbiguousMembers(ambiguousMembers)
      .addAllClassMembers(classMembers)
      .setHasDecorators(classDescriptor.hasDecorators())
      .setHasSuperClassWithoutDescriptor(classDescriptor.hasSuperClassWithoutDescriptor())
      .setHasMetaClass(classDescriptor.hasMetaClass())
      .setSupportsGenerics(classDescriptor.supportsGenerics());
    LocationInFile definitionLocation = classDescriptor.definitionLocation();
    if (definitionLocation != null) {
      builder.setDefinitionLocation(toProtobuf(definitionLocation));
    }
    String metaclassFQN = classDescriptor.metaclassFQN();
    if (metaclassFQN != null) {
      builder.setMetaClassFQN(metaclassFQN);
    }
    return builder.build();
  }

  public static DescriptorsProtos.FunctionDescriptor toProtobuf(FunctionDescriptor functionDescriptor) {
    DescriptorsProtos.FunctionDescriptor.Builder builder = DescriptorsProtos.FunctionDescriptor.newBuilder()
      .setName(functionDescriptor.name())
      .setFullyQualifiedName(functionDescriptor.fullyQualifiedName())
      .addAllParameters(functionDescriptor.parameters().stream().map(DescriptorsToProtobuf::toProtobuf).toList())
      .setIsAsynchronous(functionDescriptor.isAsynchronous())
      .setIsInstanceMethod(functionDescriptor.isInstanceMethod())
      .addAllDecorators(functionDescriptor.decorators())
      .setHasDecorators(functionDescriptor.hasDecorators());
    String annotatedReturnTypeName = functionDescriptor.annotatedReturnTypeName();
    if (annotatedReturnTypeName != null) {
      builder.setAnnotatedReturnType(annotatedReturnTypeName);
    }
    LocationInFile definitionLocation = functionDescriptor.definitionLocation();
    if (definitionLocation != null) {
      builder.setDefinitionLocation(toProtobuf(definitionLocation));
    }
    return builder.build();
  }

  public static DescriptorsProtos.ParameterDescriptor toProtobuf(FunctionDescriptor.Parameter parameterDescriptor) {
    DescriptorsProtos.ParameterDescriptor.Builder builder = DescriptorsProtos.ParameterDescriptor.newBuilder()
      .setHasDefaultValue(parameterDescriptor.hasDefaultValue())
      .setIsKeywordVariadic(parameterDescriptor.isKeywordVariadic())
      .setIsPositionalVariadic(parameterDescriptor.isPositionalVariadic())
      .setIsKeywordOnly(parameterDescriptor.isKeywordOnly())
      .setIsPositionalOnly(parameterDescriptor.isPositionalOnly());
    String annotatedType = parameterDescriptor.annotatedType();
    if (parameterDescriptor.name() != null) {
      builder.setName(parameterDescriptor.name());
    }
    if (annotatedType != null) {
      builder.setAnnotatedType(annotatedType);
    }
    LocationInFile location = parameterDescriptor.location();
    if (location != null) {
      builder.setDefinitionLocation(toProtobuf(location));
    }
    return builder.build();
  }

  public static DescriptorsProtos.VarDescriptor toProtobuf(VariableDescriptor variableDescriptor) {
    DescriptorsProtos.VarDescriptor.Builder builder = DescriptorsProtos.VarDescriptor.newBuilder();
    builder.setName(variableDescriptor.name());
    String fullyQualifiedName = variableDescriptor.fullyQualifiedName();
    if (fullyQualifiedName != null) {
      builder.setFullyQualifiedName(fullyQualifiedName);
    }
    String annotatedType = variableDescriptor.annotatedType();
    if (annotatedType != null) {
      builder.setAnnotatedType(annotatedType);
    }
    return builder.build();
  }

  public static DescriptorsProtos.LocationInFile toProtobuf(LocationInFile locationInFile) {
    return DescriptorsProtos.LocationInFile.newBuilder()
      .setFileId(locationInFile.fileId())
      .setStartLine(locationInFile.startLine())
      .setStartLineOffset(locationInFile.startLineOffset())
      .setEndLine(locationInFile.endLine())
      .setEndLineOffset(locationInFile.endLineOffset())
      .build();
  }

  public static Set<Descriptor> fromProtobuf(DescriptorsProtos.ModuleDescriptor moduleDescriptorProto) {
    Set<Descriptor> descriptors = new HashSet<>();
    moduleDescriptorProto.getClassDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    moduleDescriptorProto.getFunctionDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    moduleDescriptorProto.getAmbiguousDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    moduleDescriptorProto.getVarDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    return descriptors;
  }

  public static AmbiguousDescriptor fromProtobuf(DescriptorsProtos.AmbiguousDescriptor ambiguousDescriptor) {
    String fullyQualifiedName = ambiguousDescriptor.hasFullyQualifiedName() ? ambiguousDescriptor.getFullyQualifiedName() : null;
    Set<Descriptor> descriptors = new HashSet<>();
    ambiguousDescriptor.getClassDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    ambiguousDescriptor.getFunctionDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    ambiguousDescriptor.getVarDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    return new AmbiguousDescriptor(
      ambiguousDescriptor.getName(),
      fullyQualifiedName,
      descriptors
    );
  }

  public static ClassDescriptor fromProtobuf(DescriptorsProtos.ClassDescriptor classDescriptorProto) {
    String metaclassFQN = classDescriptorProto.hasMetaClassFQN() ? classDescriptorProto.getMetaClassFQN() : null;
    LocationInFile definitionLocation = classDescriptorProto.hasDefinitionLocation() ? fromProtobuf(classDescriptorProto.getDefinitionLocation()) : null;
    String fullyQualifiedName = classDescriptorProto.getFullyQualifiedName();
    Set<Descriptor> members = new HashSet<>();
    classDescriptorProto.getClassMembersList().forEach(proto -> members.add(fromProtobuf(proto)));
    classDescriptorProto.getFunctionMembersList().forEach(proto -> members.add(fromProtobuf(proto)));
    classDescriptorProto.getAmbiguousMembersList().forEach(proto -> members.add(fromProtobuf(proto)));
    classDescriptorProto.getVarMembersList().forEach(proto -> members.add(fromProtobuf(proto)));
    return new ClassDescriptor(
      classDescriptorProto.getName(),
      fullyQualifiedName,
      new ArrayList<>(classDescriptorProto.getSuperClassesList()),
      members,
      classDescriptorProto.getHasDecorators(),
      definitionLocation,
      classDescriptorProto.getHasSuperClassWithoutDescriptor(),
      classDescriptorProto.getHasMetaClass(),
      metaclassFQN,
      classDescriptorProto.getSupportsGenerics()
    );
  }

  public static FunctionDescriptor fromProtobuf(DescriptorsProtos.FunctionDescriptor functionDescriptorProto) {
    String fullyQualifiedName = functionDescriptorProto.getFullyQualifiedName();
    List<FunctionDescriptor.Parameter> parameters = new ArrayList<>();
    functionDescriptorProto.getParametersList().forEach(proto -> parameters.add(fromProtobuf(proto)));
    LocationInFile definitionLocation = functionDescriptorProto.hasDefinitionLocation() ? fromProtobuf(functionDescriptorProto.getDefinitionLocation()) : null;
    String annotatedReturnTypeName = functionDescriptorProto.hasAnnotatedReturnType() ? functionDescriptorProto.getAnnotatedReturnType() : null;
    return new FunctionDescriptor(
      functionDescriptorProto.getName(),
      fullyQualifiedName,
      parameters,
      functionDescriptorProto.getIsAsynchronous(),
      functionDescriptorProto.getIsInstanceMethod(),
      new ArrayList<>(functionDescriptorProto.getDecoratorsList()),
      functionDescriptorProto.getHasDecorators(),
      definitionLocation,
      annotatedReturnTypeName
    );
  }

  public static FunctionDescriptor.Parameter fromProtobuf(DescriptorsProtos.ParameterDescriptor parameterDescriptorProto) {
    String name = parameterDescriptorProto.hasName() ? parameterDescriptorProto.getName() : null;
    String annotatedType = parameterDescriptorProto.hasAnnotatedType() ? parameterDescriptorProto.getAnnotatedType() : null;
    LocationInFile location = parameterDescriptorProto.hasDefinitionLocation() ? fromProtobuf(parameterDescriptorProto.getDefinitionLocation()) : null;
    return new FunctionDescriptor.Parameter(
      name,
      annotatedType,
      parameterDescriptorProto.getHasDefaultValue(),
      parameterDescriptorProto.getIsKeywordOnly(),
      parameterDescriptorProto.getIsPositionalOnly(),
      parameterDescriptorProto.getIsPositionalVariadic(),
      parameterDescriptorProto.getIsKeywordVariadic(),
      location
    );
  }

  public static VariableDescriptor fromProtobuf(DescriptorsProtos.VarDescriptor varDescriptorProto) {
    String fullyQualifiedName = varDescriptorProto.hasFullyQualifiedName() ? varDescriptorProto.getFullyQualifiedName() : null;
    String annotatedType = varDescriptorProto.hasAnnotatedType() ? varDescriptorProto.getAnnotatedType() : null;
    return new VariableDescriptor(
      varDescriptorProto.getName(),
      fullyQualifiedName,
      annotatedType
    );
  }

  public static LocationInFile fromProtobuf(DescriptorsProtos.LocationInFile locationInFileProto) {
    return new LocationInFile(
      locationInFileProto.getFileId(),
      locationInFileProto.getStartLine(),
      locationInFileProto.getStartLineOffset(),
      locationInFileProto.getEndLine(),
      locationInFileProto.getEndLineOffset()
    );
  }
}
