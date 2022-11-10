/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.index;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class ClassDescriptor implements Descriptor {

  private final String name;
  @Nullable
  private final String fullyQualifiedName;
  private final Collection<String> superClasses;
  private final Collection<Descriptor> members;
  private final boolean hasDecorators;
  private final LocationInFile definitionLocation;
  private final boolean hasSuperClassWithoutDescriptor;
  private final boolean hasMetaClass;
  private final String metaclassFQN;
  private final boolean supportsGenerics;

  public ClassDescriptor(String name, @Nullable String fullyQualifiedName, Collection<String> superClasses, Collection<Descriptor> members,
    boolean hasDecorators, @Nullable LocationInFile definitionLocation, boolean hasSuperClassWithoutDescriptor, boolean hasMetaClass,
    @Nullable String metaclassFQN, boolean supportsGenerics) {

    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.superClasses = superClasses;
    this.members = members;
    this.hasDecorators = hasDecorators;
    this.definitionLocation = definitionLocation;
    this.hasSuperClassWithoutDescriptor = hasSuperClassWithoutDescriptor;
    this.hasMetaClass = hasMetaClass;
    this.metaclassFQN = metaclassFQN;
    this.supportsGenerics = supportsGenerics;
  }

  public ClassDescriptor(DescriptorsProtos.ClassDescriptor classDescriptorProto) {
    name = classDescriptorProto.getName();
    fullyQualifiedName = classDescriptorProto.getFullyQualifiedName();
    superClasses = new ArrayList<>(classDescriptorProto.getSuperClassesList());
    definitionLocation = new LocationInFile(classDescriptorProto.getDefinitionLocation());
    hasDecorators = classDescriptorProto.getHasDecorators();
    hasSuperClassWithoutDescriptor = classDescriptorProto.getHasSuperClassWithoutDescriptor();
    hasMetaClass = classDescriptorProto.getHasMetaClass();
    metaclassFQN = classDescriptorProto.hasMetaClassFQN() ? classDescriptorProto.getMetaClassFQN() : null;
    supportsGenerics = classDescriptorProto.getSupportsGenerics();
    members = new ArrayList<>();
    classDescriptorProto.getClassMembersList().forEach(proto -> members.add(new ClassDescriptor(proto)));
    classDescriptorProto.getFunctionMembersList().forEach(proto -> members.add(new FunctionDescriptor(proto)));
    classDescriptorProto.getAmbiguousMembersList().forEach(proto -> members.add(new AmbiguousDescriptor(proto)));
    classDescriptorProto.getVarMembersList().forEach(proto -> members.add(new VariableDescriptor(proto)));
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @Override
  public Kind kind() {
    return Kind.CLASS;
  }

  public Collection<String> superClasses() {
    return superClasses;
  }

  public Collection<Descriptor> members() {
    return members;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  public boolean hasSuperClassWithoutDescriptor() {
    return hasSuperClassWithoutDescriptor;
  }

  public LocationInFile definitionLocation() {
    return definitionLocation;
  }

  public boolean hasMetaClass() {
    return hasMetaClass;
  }

  public String metaclassFQN() {
    return metaclassFQN;
  }

  public boolean supportsGenerics() {
    return supportsGenerics;
  }

  public static class ClassDescriptorBuilder {

    private String name;
    private String fullyQualifiedName;
    private Collection<String> superClasses = new HashSet<>();
    private Collection<Descriptor> members = new HashSet<>();
    private boolean hasDecorators = false;
    private LocationInFile definitionLocation = null;
    private boolean hasSuperClassWithoutDescriptor = false;
    private boolean hasMetaClass = false;
    private String metaclassFQN = null;
    private boolean supportsGenerics = false;

    public ClassDescriptorBuilder withName(String name) {
      this.name = name;
      return this;
    }

    public ClassDescriptorBuilder withFullyQualifiedName(@Nullable String fullyQualifiedName) {
      this.fullyQualifiedName = fullyQualifiedName;
      return this;
    }

    public ClassDescriptorBuilder withSuperClasses(Collection<String> superClasses) {
      this.superClasses = superClasses;
      return this;
    }

    public ClassDescriptorBuilder withMembers(Collection<Descriptor> members) {
      this.members = members;
      return this;
    }

    public ClassDescriptorBuilder withHasDecorators(boolean hasDecorators) {
      this.hasDecorators = hasDecorators;
      return this;
    }

    public ClassDescriptorBuilder withHasSuperClassWithoutDescriptor(boolean hasSuperClassWithoutDescriptor) {
      this.hasSuperClassWithoutDescriptor = hasSuperClassWithoutDescriptor;
      return this;
    }

    public ClassDescriptorBuilder withDefinitionLocation(@Nullable LocationInFile definitionLocation) {
      this.definitionLocation = definitionLocation;
      return this;
    }

    public ClassDescriptorBuilder withHasMetaClass(boolean hasMetaClass) {
      this.hasMetaClass = hasMetaClass;
      return this;
    }

    public ClassDescriptorBuilder withMetaclassFQN(@Nullable String metaclassFQN) {
      this.metaclassFQN = metaclassFQN;
      return this;
    }

    public ClassDescriptorBuilder withSupportsGenerics(boolean supportsGenerics) {
      this.supportsGenerics = supportsGenerics;
      return this;
    }

    public ClassDescriptor build() {
      return new ClassDescriptor(name, fullyQualifiedName, superClasses, members, hasDecorators, definitionLocation,
        hasSuperClassWithoutDescriptor, hasMetaClass, metaclassFQN, supportsGenerics);
    }
  }

  public DescriptorsProtos.ClassDescriptor toProtobuf() {
    List<DescriptorsProtos.FunctionDescriptor> functionMembers = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> variableMembers = new ArrayList<>();
    List<DescriptorsProtos.AmbiguousDescriptor> ambiguousMembers = new ArrayList<>();
    List<DescriptorsProtos.ClassDescriptor> classMembers = new ArrayList<>();
    for (Descriptor member : members) {
      if (member.kind().equals(Kind.FUNCTION)) {
        functionMembers.add(((FunctionDescriptor) member).toProtobuf());
      } else if (member.kind().equals(Kind.VARIABLE)) {
        variableMembers.add(((VariableDescriptor) member).toProtobuf());
      } else if (member.kind().equals(Kind.AMBIGUOUS)) {
        ambiguousMembers.add(((AmbiguousDescriptor) member).toProtobuf());
      } else {
        classMembers.add(((ClassDescriptor) member).toProtobuf());
      }
    }
    DescriptorsProtos.ClassDescriptor.Builder builder = DescriptorsProtos.ClassDescriptor.newBuilder()
      .setName(name)
      .setFullyQualifiedName(fullyQualifiedName)
      .addAllSuperClasses(superClasses)
      .addAllFunctionMembers(functionMembers)
      .addAllVarMembers(variableMembers)
      .addAllAmbiguousMembers(ambiguousMembers)
      .addAllClassMembers(classMembers)
      .setHasDecorators(hasDecorators)
      .setHasSuperClassWithoutDescriptor(hasSuperClassWithoutDescriptor)
      .setHasMetaClass(hasMetaClass)
      .setSupportsGenerics(supportsGenerics);
    if (definitionLocation != null) {
      builder.setDefinitionLocation(definitionLocation.toProtobuf());
    }
    if (metaclassFQN != null) {
      builder.setMetaClassFQN(metaclassFQN);
    }
    return builder.build();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ClassDescriptor that = (ClassDescriptor) o;
    return hasDecorators == that.hasDecorators &&
      hasSuperClassWithoutDescriptor == that.hasSuperClassWithoutDescriptor &&
      hasMetaClass == that.hasMetaClass &&
      supportsGenerics == that.supportsGenerics &&
      name.equals(that.name) &&
      Objects.equals(fullyQualifiedName, that.fullyQualifiedName) &&
      superClasses.equals(that.superClasses) &&
      members.equals(that.members) &&
      Objects.equals(definitionLocation, that.definitionLocation) &&
      Objects.equals(metaclassFQN, that.metaclassFQN);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, fullyQualifiedName, superClasses, members, hasDecorators, definitionLocation, hasSuperClassWithoutDescriptor, hasMetaClass, metaclassFQN, supportsGenerics);
  }
}
