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

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;

public class ClassDescriptor implements Descriptor {

  private final String name;
  private final String fullyQualifiedName;
  private final Collection<String> superClasses;
  private final Set<Descriptor> members;
  private final boolean hasDecorators;
  private final LocationInFile definitionLocation;
  private final boolean hasSuperClassWithoutDescriptor;
  private final boolean hasMetaClass;
  private final String metaclassFQN;
  private final boolean supportsGenerics;

  public ClassDescriptor(String name, String fullyQualifiedName, Collection<String> superClasses, Set<Descriptor> members,
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

  @Override
  public String name() {
    return name;
  }

  @Override
  @Nonnull
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

  @CheckForNull
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
    private Set<Descriptor> members = new HashSet<>();
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

    public ClassDescriptorBuilder withFullyQualifiedName(String fullyQualifiedName) {
      this.fullyQualifiedName = fullyQualifiedName;
      return this;
    }

    public ClassDescriptorBuilder withSuperClasses(Collection<String> superClasses) {
      this.superClasses = superClasses;
      return this;
    }

    public ClassDescriptorBuilder withMembers(Set<Descriptor> members) {
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
}
