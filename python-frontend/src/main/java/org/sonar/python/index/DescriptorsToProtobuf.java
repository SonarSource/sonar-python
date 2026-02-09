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
package org.sonar.python.index;

import com.google.common.annotations.VisibleForTesting;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.types.protobuf.DescriptorsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class DescriptorsToProtobuf {
  private static final Logger LOG = LoggerFactory.getLogger(DescriptorsToProtobuf.class);

  private DescriptorsToProtobuf() {
  }

  public static DescriptorsProtos.ModuleDescriptor toProtobufModuleDescriptor(Set<Descriptor> descriptors) {
    return DescriptorsProtos.ModuleDescriptor.newBuilder()
      .setDescriptors(toProtobufDescriptorList(descriptors))
      .build();
  }

  private static DescriptorsProtos.DescriptorList toProtobufDescriptorList(Collection<Descriptor> descriptors) {
    List<DescriptorsProtos.ClassDescriptor> classDescriptors = new ArrayList<>();
    List<DescriptorsProtos.FunctionDescriptor> functionDescriptors = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> varDescriptors = new ArrayList<>();
    List<DescriptorsProtos.AmbiguousDescriptor> ambiguousDescriptors = new ArrayList<>();
    for (Descriptor descriptor : descriptors) {
      switch (descriptor.kind()) {
        case CLASS -> classDescriptors.add(toProtobuf((ClassDescriptor) descriptor));
        case FUNCTION -> functionDescriptors.add(toProtobuf((FunctionDescriptor) descriptor));
        case VARIABLE -> varDescriptors.add(toProtobuf((VariableDescriptor) descriptor));
        case AMBIGUOUS -> ambiguousDescriptors.add(toProtobuf((AmbiguousDescriptor) descriptor));
        default -> LOG.debug("Unknown descriptor kind: {}", descriptor.kind());
      }
    }
    return DescriptorsProtos.DescriptorList.newBuilder()
      .addAllClassDescriptors(classDescriptors)
      .addAllFunctionDescriptors(functionDescriptors)
      .addAllVarDescriptors(varDescriptors)
      .addAllAmbiguousDescriptors(ambiguousDescriptors)
      .build();
  }

  private static DescriptorsProtos.AmbiguousDescriptor toProtobuf(AmbiguousDescriptor ambiguousDescriptor) {
    DescriptorsProtos.AmbiguousDescriptor.Builder builder = DescriptorsProtos.AmbiguousDescriptor.newBuilder()
      .setName(ambiguousDescriptor.name())
      .setAlternatives(toProtobufDescriptorList(ambiguousDescriptor.alternatives()));
    String fullyQualifiedName = ambiguousDescriptor.fullyQualifiedName();
    if (fullyQualifiedName != null) {
      builder.setFullyQualifiedName(fullyQualifiedName);
    }
    return builder.build();
  }

  private static DescriptorsProtos.ClassDescriptor toProtobuf(ClassDescriptor classDescriptor) {
    DescriptorsProtos.ClassDescriptor.Builder builder = DescriptorsProtos.ClassDescriptor.newBuilder()
      .setName(classDescriptor.name())
      .setFullyQualifiedName(classDescriptor.fullyQualifiedName())
      .addAllSuperClasses(classDescriptor.superClasses())
      .setMembers(toProtobufDescriptorList(classDescriptor.members()))
      .setAttributes(toProtobufDescriptorList(classDescriptor.attributes()))
      .setHasDecorators(classDescriptor.hasDecorators())
      .setHasSuperClassWithoutDescriptor(classDescriptor.hasSuperClassWithoutDescriptor())
      .setHasMetaClass(classDescriptor.hasMetaClass())
      .setMetaClasses(toProtobufDescriptorList(classDescriptor.metaClasses()))
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

  private static DescriptorsProtos.FunctionDescriptor toProtobuf(FunctionDescriptor functionDescriptor) {
    DescriptorsProtos.FunctionDescriptor.Builder builder = DescriptorsProtos.FunctionDescriptor.newBuilder()
      .setName(functionDescriptor.name())
      .setFullyQualifiedName(functionDescriptor.fullyQualifiedName())
      .addAllParameters(functionDescriptor.parameters().stream().map(DescriptorsToProtobuf::toProtobuf).toList())
      .setIsAsynchronous(functionDescriptor.isAsynchronous())
      .setIsInstanceMethod(functionDescriptor.isInstanceMethod())
      .setIsClassMethod(functionDescriptor.isClassMethod())
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
    TypeAnnotationDescriptor returnTypeAnnotationDescriptor = functionDescriptor.typeAnnotationDescriptor();
    if (returnTypeAnnotationDescriptor != null) {
      builder.setReturnTypeAnnotationDescriptor(toProtobuf(returnTypeAnnotationDescriptor));
    }
    return builder.build();
  }

  private static DescriptorsProtos.ParameterDescriptor toProtobuf(FunctionDescriptor.Parameter parameterDescriptor) {
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
    TypeAnnotationDescriptor typeDescriptor = parameterDescriptor.descriptor();
    if (typeDescriptor != null) {
      builder.setTypeAnnotationDescriptor(toProtobuf(typeDescriptor));
    }
    LocationInFile location = parameterDescriptor.location();
    if (location != null) {
      builder.setDefinitionLocation(toProtobuf(location));
    }
    return builder.build();
  }

  @VisibleForTesting
  static DescriptorsProtos.VarDescriptor toProtobuf(VariableDescriptor variableDescriptor) {
    DescriptorsProtos.VarDescriptor.Builder builder = DescriptorsProtos.VarDescriptor.newBuilder()
      .setName(variableDescriptor.name())
      .setAttributes(toProtobufDescriptorList(variableDescriptor.attributes()))
      .setMembers(toProtobufDescriptorList(variableDescriptor.members()));
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

  private static DescriptorsProtos.LocationInFile toProtobuf(LocationInFile locationInFile) {
    return DescriptorsProtos.LocationInFile.newBuilder()
      .setFileId(locationInFile.fileId())
      .setStartLine(locationInFile.startLine())
      .setStartLineOffset(locationInFile.startLineOffset())
      .setEndLine(locationInFile.endLine())
      .setEndLineOffset(locationInFile.endLineOffset())
      .build();
  }

  private static SymbolsProtos.Type toProtobuf(TypeAnnotationDescriptor typeAnnotationDescriptor) {
    SymbolsProtos.Type.Builder builder = SymbolsProtos.Type.newBuilder()
      .setPrettyPrintedName(typeAnnotationDescriptor.prettyPrintedName())
      .setKind(DescriptorUtils.typeAnnotationKindToSymbolKind(typeAnnotationDescriptor.kind()))
      .addAllArgs(typeAnnotationDescriptor.args().stream().map(DescriptorsToProtobuf::toProtobuf).toList())
      .setIsSelf(typeAnnotationDescriptor.isSelf());
    String fullyQualifiedName = typeAnnotationDescriptor.fullyQualifiedName();
    if (fullyQualifiedName != null) {
      builder.setFullyQualifiedName(fullyQualifiedName);
    }
    return builder.build();
  }

  public static Set<Descriptor> fromProtobuf(DescriptorsProtos.ModuleDescriptor moduleDescriptorProto) {
    return fromProtobufDescriptorList(moduleDescriptorProto.getDescriptors());
  }

  private static Set<Descriptor> fromProtobufDescriptorList(DescriptorsProtos.DescriptorList descriptorList) {
    Set<Descriptor> descriptors = new HashSet<>();
    descriptorList.getClassDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    descriptorList.getFunctionDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    descriptorList.getAmbiguousDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    descriptorList.getVarDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    return descriptors;
  }

  private static List<Descriptor> fromProtobufDescriptorListAsList(DescriptorsProtos.DescriptorList descriptorList) {
    List<Descriptor> descriptors = new ArrayList<>();
    descriptorList.getClassDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    descriptorList.getFunctionDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    descriptorList.getAmbiguousDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    descriptorList.getVarDescriptorsList().forEach(proto -> descriptors.add(fromProtobuf(proto)));
    return descriptors;
  }

  private static AmbiguousDescriptor fromProtobuf(DescriptorsProtos.AmbiguousDescriptor ambiguousDescriptor) {
    String fullyQualifiedName = ambiguousDescriptor.hasFullyQualifiedName() ? ambiguousDescriptor.getFullyQualifiedName() : null;
    Set<Descriptor> alternatives = fromProtobufDescriptorList(ambiguousDescriptor.getAlternatives());
    return new AmbiguousDescriptor(
      ambiguousDescriptor.getName(),
      fullyQualifiedName,
      alternatives);
  }

  private static ClassDescriptor fromProtobuf(DescriptorsProtos.ClassDescriptor classDescriptorProto) {
    String metaclassFQN = classDescriptorProto.hasMetaClassFQN() ? classDescriptorProto.getMetaClassFQN() : null;
    LocationInFile definitionLocation = classDescriptorProto.hasDefinitionLocation() ? fromProtobuf(classDescriptorProto.getDefinitionLocation()) : null;
    String fullyQualifiedName = classDescriptorProto.getFullyQualifiedName();
    Set<Descriptor> members = fromProtobufDescriptorList(classDescriptorProto.getMembers());
    List<Descriptor> attributes = fromProtobufDescriptorListAsList(classDescriptorProto.getAttributes());
    List<Descriptor> metaClasses = fromProtobufDescriptorListAsList(classDescriptorProto.getMetaClasses());
    return new ClassDescriptor.ClassDescriptorBuilder()
      .withName(classDescriptorProto.getName())
      .withFullyQualifiedName(fullyQualifiedName)
      .withSuperClasses(new ArrayList<>(classDescriptorProto.getSuperClassesList()))
      .withMembers(members)
      .withAttributes(attributes)
      .withHasDecorators(classDescriptorProto.getHasDecorators())
      .withDefinitionLocation(definitionLocation)
      .withHasSuperClassWithoutDescriptor(classDescriptorProto.getHasSuperClassWithoutDescriptor())
      .withHasMetaClass(classDescriptorProto.getHasMetaClass())
      .withMetaclassFQN(metaclassFQN)
      .withMetaClasses(metaClasses)
      .withSupportsGenerics(classDescriptorProto.getSupportsGenerics())
      .build();
  }

  private static FunctionDescriptor fromProtobuf(DescriptorsProtos.FunctionDescriptor functionDescriptorProto) {
    String fullyQualifiedName = functionDescriptorProto.getFullyQualifiedName();
    List<FunctionDescriptor.Parameter> parameters = new ArrayList<>();
    functionDescriptorProto.getParametersList().forEach(proto -> parameters.add(fromProtobuf(proto)));
    LocationInFile definitionLocation = functionDescriptorProto.hasDefinitionLocation() ? fromProtobuf(functionDescriptorProto.getDefinitionLocation()) : null;
    String annotatedReturnTypeName = functionDescriptorProto.hasAnnotatedReturnType() ? functionDescriptorProto.getAnnotatedReturnType() : null;
    TypeAnnotationDescriptor returnTypeAnnotationDescriptor = functionDescriptorProto.hasReturnTypeAnnotationDescriptor()
      ? fromProtobuf(functionDescriptorProto.getReturnTypeAnnotationDescriptor())
      : null;
    return new FunctionDescriptor.FunctionDescriptorBuilder()
      .withName(functionDescriptorProto.getName())
      .withFullyQualifiedName(fullyQualifiedName)
      .withParameters(parameters)
      .withIsAsynchronous(functionDescriptorProto.getIsAsynchronous())
      .withIsInstanceMethod(functionDescriptorProto.getIsInstanceMethod())
      .withIsClassMethod(functionDescriptorProto.getIsClassMethod())
      .withDecorators(new ArrayList<>(functionDescriptorProto.getDecoratorsList()))
      .withHasDecorators(functionDescriptorProto.getHasDecorators())
      .withDefinitionLocation(definitionLocation)
      .withAnnotatedReturnTypeName(annotatedReturnTypeName)
      .withTypeAnnotationDescriptor(returnTypeAnnotationDescriptor)
      .build();
  }

  private static FunctionDescriptor.Parameter fromProtobuf(DescriptorsProtos.ParameterDescriptor parameterDescriptorProto) {
    String name = parameterDescriptorProto.hasName() ? parameterDescriptorProto.getName() : null;
    String annotatedType = parameterDescriptorProto.hasAnnotatedType() ? parameterDescriptorProto.getAnnotatedType() : null;
    LocationInFile location = parameterDescriptorProto.hasDefinitionLocation() ? fromProtobuf(parameterDescriptorProto.getDefinitionLocation()) : null;
    TypeAnnotationDescriptor typeAnnotationDescriptor = parameterDescriptorProto.hasTypeAnnotationDescriptor()
      ? fromProtobuf(parameterDescriptorProto.getTypeAnnotationDescriptor())
      : null;
    return new FunctionDescriptor.Parameter(
      name,
      annotatedType,
      typeAnnotationDescriptor,
      parameterDescriptorProto.getHasDefaultValue(),
      parameterDescriptorProto.getIsKeywordOnly(),
      parameterDescriptorProto.getIsPositionalOnly(),
      parameterDescriptorProto.getIsPositionalVariadic(),
      parameterDescriptorProto.getIsKeywordVariadic(),
      location);
  }

  @VisibleForTesting
  static VariableDescriptor fromProtobuf(DescriptorsProtos.VarDescriptor varDescriptorProto) {
    String fullyQualifiedName = varDescriptorProto.hasFullyQualifiedName() ? varDescriptorProto.getFullyQualifiedName() : null;
    String annotatedType = varDescriptorProto.hasAnnotatedType() ? varDescriptorProto.getAnnotatedType() : null;
    List<Descriptor> attributes = fromProtobufDescriptorListAsList(varDescriptorProto.getAttributes());
    List<Descriptor> members = fromProtobufDescriptorListAsList(varDescriptorProto.getMembers());
    return new VariableDescriptor(
      varDescriptorProto.getName(),
      fullyQualifiedName,
      annotatedType,
      false,
      attributes,
      members);
  }

  private static LocationInFile fromProtobuf(DescriptorsProtos.LocationInFile locationInFileProto) {
    return new LocationInFile(
      locationInFileProto.getFileId(),
      locationInFileProto.getStartLine(),
      locationInFileProto.getStartLineOffset(),
      locationInFileProto.getEndLine(),
      locationInFileProto.getEndLineOffset());
  }

  @CheckForNull
  private static TypeAnnotationDescriptor fromProtobuf(SymbolsProtos.Type typeProto) {
    String fullyQualifiedName = typeProto.hasFullyQualifiedName() ? typeProto.getFullyQualifiedName() : null;
    var kind = DescriptorUtils.symbolTypeKindToTypeAnnotationKind(typeProto.getKind());
    if (kind == null) {
      return null;
    }
    List<TypeAnnotationDescriptor> args = typeProto.getArgsList().stream()
      .map(DescriptorsToProtobuf::fromProtobuf)
      .filter(Objects::nonNull)
      .toList();

    return new TypeAnnotationDescriptor(
      typeProto.getPrettyPrintedName(),
      kind,
      args,
      fullyQualifiedName,
      typeProto.getIsSelf());
  }
}
