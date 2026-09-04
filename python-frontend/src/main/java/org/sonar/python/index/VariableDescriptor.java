/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public class VariableDescriptor implements Descriptor {
  private final String name;
  private final String fullyQualifiedName;
  private final String annotatedType;
  private final boolean isImportedModule;
  private final boolean isTypeAlias;
  private final List<Descriptor> attributes;
  private final List<Descriptor> members;

  public VariableDescriptor(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedType,
                            boolean isImportedModule, List<Descriptor> attributes, List<Descriptor> members) {
    this(name, fullyQualifiedName, annotatedType, isImportedModule, false, attributes, members);
  }

  public VariableDescriptor(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedType,
                            boolean isImportedModule, boolean isTypeAlias, List<Descriptor> attributes, List<Descriptor> members) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.annotatedType = annotatedType;
    this.isImportedModule = isImportedModule;
    this.isTypeAlias = isTypeAlias;
    this.attributes = attributes;
    this.members = members;
  }

  public VariableDescriptor(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedType) {
    this(name, fullyQualifiedName, annotatedType, false, false, List.of(), List.of());
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

  public boolean isTypeAlias() {
    return isTypeAlias;
  }

  public List<Descriptor> attributes() {
    return attributes;
  }

  public List<Descriptor> members() {
    return members;
  }
}
