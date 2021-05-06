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

public class ModuleSummary implements Summary {

  private final String name;
  private final String fullyQualifiedName;
  private final Collection<FunctionSummary> functions = new HashSet<>();
  private final Collection<ClassSummary> classes = new HashSet<>();
  private final Collection<VariableSummary> variables = new HashSet<>();
  private final Map<String, Collection<Summary>> summariesByFQN = new HashMap<>();

  public ModuleSummary(String name, String fullyQualifiedName, Collection<Summary> summaries) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    for (Summary summary : summaries) {
      if (summary.fullyQualifiedName() != null) {
        summariesByFQN.computeIfAbsent(summary.fullyQualifiedName(), k -> new HashSet<>()).add(summary);
      }
      if (summary instanceof FunctionSummary) {
        this.functions.add((FunctionSummary) summary);
      }
      if (summary instanceof ClassSummary) {
        this.classes.add((ClassSummary) summary);
      }
      if (summary instanceof VariableSummary) {
        this.variables.add((VariableSummary) summary);
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

  public Collection<FunctionSummary> functions() {
    return functions;
  }

  public Collection<ClassSummary> classes() {
    return classes;
  }

  public Collection<VariableSummary> variables() {
    return variables;
  }

  public Collection<Summary> summariesWithFQN(@Nullable String fullyQualifiedName) {
    return summariesByFQN.getOrDefault(fullyQualifiedName, Collections.emptySet());
  }

}
