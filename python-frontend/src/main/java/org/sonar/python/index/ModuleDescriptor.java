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

import java.util.Map;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public class ModuleDescriptor implements Descriptor{
  private final String name;
  private final String fullyQualifiedName;
  private final Map<String, Descriptor> members;

  public ModuleDescriptor(String name, @Nullable String fullyQualifiedName, Map<String, Descriptor> members) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.members = members;
  }

  @Override
  public String name() {
    return name;
  }

  @CheckForNull
  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public Map<String, Descriptor> members() {
    return members;
  }

  @Override
  public Kind kind() {
    return Kind.MODULE;
  }
}
