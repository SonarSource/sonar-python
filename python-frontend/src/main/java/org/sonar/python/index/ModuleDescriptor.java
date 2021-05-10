/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import javax.annotation.Nullable;

public class ModuleDescriptor implements Descriptor {

  private final String name;
  private final String fullyQualifiedName;
  private final Collection<FunctionDescriptor> functions = new HashSet<>();
  private final Collection<ClassDescriptor> classes = new HashSet<>();
  private final Collection<VariableDescriptor> variables = new HashSet<>();
  private final Map<String, Collection<Descriptor>> descriptorsByFQN = new HashMap<>();

  public ModuleDescriptor(String name, String fullyQualifiedName, Collection<Descriptor> descriptors) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    for (Descriptor descriptor : descriptors) {
      if (descriptor.fullyQualifiedName() != null) {
        descriptorsByFQN.computeIfAbsent(descriptor.fullyQualifiedName(), k -> new HashSet<>()).add(descriptor);
      }
      if (descriptor instanceof FunctionDescriptor) {
        this.functions.add((FunctionDescriptor) descriptor);
      }
      if (descriptor instanceof ClassDescriptor) {
        this.classes.add((ClassDescriptor) descriptor);
      }
      if (descriptor instanceof VariableDescriptor) {
        this.variables.add((VariableDescriptor) descriptor);
      }
    }
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
    return Kind.MODULE;
  }

  public Collection<FunctionDescriptor> functions() {
    return functions;
  }

  public Collection<ClassDescriptor> classes() {
    return classes;
  }

  public Collection<VariableDescriptor> variables() {
    return variables;
  }

  public Collection<Descriptor> descriptorsWithFQN(@Nullable String fullyQualifiedName) {
    return descriptorsByFQN.getOrDefault(fullyQualifiedName, Collections.emptySet());
  }

}
