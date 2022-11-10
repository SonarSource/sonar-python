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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class AmbiguousDescriptor implements Descriptor {

  private final Set<Descriptor> descriptors;
  private final String name;
  private final String fullyQualifiedName;

  public AmbiguousDescriptor(String name, @Nullable String fullyQualifiedName, Set<Descriptor> descriptors) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.descriptors = descriptors;
  }

  public AmbiguousDescriptor(DescriptorsProtos.AmbiguousDescriptor ambiguousDescriptor) {
    this.name = ambiguousDescriptor.getName();
    this.fullyQualifiedName = ambiguousDescriptor.getFullyQualifiedName();
    descriptors = new HashSet<>();
    ambiguousDescriptor.getClassDescriptorsList().forEach(proto -> descriptors.add(new ClassDescriptor(proto)));
    ambiguousDescriptor.getFunctionDescriptorsList().forEach(proto -> descriptors.add(new FunctionDescriptor(proto)));
    ambiguousDescriptor.getVarDescriptorsList().forEach(proto -> descriptors.add(new VariableDescriptor(proto)));
  }

  @Override
  public String name() {
    return name;
  }

  @Nullable
  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @Override
  public Kind kind() {
    return Kind.AMBIGUOUS;
  }

  public Set<Descriptor> alternatives() {
    return descriptors;
  }

  public static AmbiguousDescriptor create(Descriptor... descriptors) {
    return create(new HashSet<>(Arrays.asList(descriptors)));
  }

  public static AmbiguousDescriptor create(Set<Descriptor> descriptors) {
    if (descriptors.size() < 2) {
      throw new IllegalArgumentException("Ambiguous symbol should contain at least two descriptors");
    }
    Descriptor firstSymbol = descriptors.iterator().next();
    String resultingSymbolName = firstSymbol.name();
    if (!descriptors.stream().map(Descriptor::name).allMatch(symbolName -> symbolName.equals(firstSymbol.name()))) {
      throw new IllegalArgumentException("Ambiguous descriptor should contain descriptors with the same name.");
    }
    return new AmbiguousDescriptor(resultingSymbolName, firstSymbol.fullyQualifiedName(), flattenAmbiguousDescriptors(descriptors));
  }

  private static Set<Descriptor> flattenAmbiguousDescriptors(Set<Descriptor> descriptors) {
    Set<Descriptor> alternatives = new HashSet<>();
    for (Descriptor descriptor : descriptors) {
      if (descriptor.kind() == Kind.AMBIGUOUS) {
        Set<Descriptor> flattenedAlternatives = flattenAmbiguousDescriptors(((AmbiguousDescriptor) descriptor).alternatives());
        alternatives.addAll(flattenedAlternatives);
      } else {
        alternatives.add(descriptor);
      }
    }
    return alternatives;
  }

  public DescriptorsProtos.AmbiguousDescriptor toProtobuf() {
    List<DescriptorsProtos.FunctionDescriptor> functionDescriptors = new ArrayList<>();
    List<DescriptorsProtos.VarDescriptor> variableDescriptors = new ArrayList<>();
    List<DescriptorsProtos.ClassDescriptor> classDescriptors = new ArrayList<>();
    for (Descriptor descriptor : descriptors) {
      if (descriptor.kind() == Kind.FUNCTION) {
        functionDescriptors.add(((FunctionDescriptor) descriptor).toProtobuf());
      } else if (descriptor.kind() == Kind.VARIABLE) {
        variableDescriptors.add(((VariableDescriptor) descriptor).toProtobuf());
      } else if (descriptor.kind() == Kind.CLASS) {
        classDescriptors.add(((ClassDescriptor) descriptor).toProtobuf());
      }
    }
    return DescriptorsProtos.AmbiguousDescriptor.newBuilder()
      .setName(name)
      .setFullyQualifiedName(fullyQualifiedName)
      .addAllClassDescriptors(classDescriptors)
      .addAllFunctionDescriptors(functionDescriptors)
      .addAllVarDescriptors(variableDescriptors)
      .build();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    AmbiguousDescriptor that = (AmbiguousDescriptor) o;
    return descriptors.equals(that.descriptors) && name.equals(that.name) && fullyQualifiedName.equals(that.fullyQualifiedName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(descriptors, name, fullyQualifiedName);
  }
}
