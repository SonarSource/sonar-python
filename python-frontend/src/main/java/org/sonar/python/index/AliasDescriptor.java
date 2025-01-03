/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import javax.annotation.Nonnull;

public class AliasDescriptor implements Descriptor {

  private final String name;
  private final String fullyQualifiedName;
  private final Descriptor originalDescriptor;

  public AliasDescriptor(String name, String fullyQualifiedName, Descriptor originalDescriptor) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.originalDescriptor = originalDescriptor;
  }

  @Override
  public String name() {
    return this.name;
  }

  @Override
  @Nonnull
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public Descriptor originalDescriptor() {
    return this.originalDescriptor;
  }

  @Override
  public Kind kind() {
    return Kind.ALIAS;
  }
}
