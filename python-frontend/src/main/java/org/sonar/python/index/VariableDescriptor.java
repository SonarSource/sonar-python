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

import java.util.Objects;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class VariableDescriptor implements Descriptor {
  private final String name;
  private final String fullyQualifiedName;
  private final String annotatedType;

  public VariableDescriptor(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedType) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.annotatedType = annotatedType;
  }

  public VariableDescriptor(DescriptorsProtos.VarDescriptor varDescriptorProto) {
    this.name = varDescriptorProto.getName();
    this.fullyQualifiedName = varDescriptorProto.hasFullyQualifiedName() ? varDescriptorProto.getFullyQualifiedName() : null;
    this.annotatedType = varDescriptorProto.hasAnnotatedType() ? varDescriptorProto.getAnnotatedType() : null;
  }

  public DescriptorsProtos.VarDescriptor toProtobuf() {
    return DescriptorsProtos.VarDescriptor.newBuilder()
      .setName(name)
      .setFullyQualifiedName(fullyQualifiedName)
      .setAnnotatedType(annotatedType)
      .build();
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
    return Kind.VARIABLE;
  }

  @CheckForNull
  public String annotatedType() {
    return annotatedType;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    VariableDescriptor that = (VariableDescriptor) o;
    return name.equals(that.name) && Objects.equals(fullyQualifiedName, that.fullyQualifiedName) && Objects.equals(annotatedType, that.annotatedType);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, fullyQualifiedName, annotatedType);
  }
}
