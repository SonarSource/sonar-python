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

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public class VariableDescriptor implements Descriptor {
  private final String name;
  private final String fullyQualifiedName;
  private final String annotatedType;
  private final boolean isImportedModule;

  public VariableDescriptor(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedType, boolean isImportedModule) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.annotatedType = annotatedType;
    this.isImportedModule = isImportedModule;
  }

  public VariableDescriptor(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedType) {
    this(name, fullyQualifiedName, annotatedType, false);
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

  public boolean isImportedModule() {
    return isImportedModule;
  }
}
