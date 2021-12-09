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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

public class AmbiguousDescriptor implements Descriptor {

  private final Set<Descriptor> descriptors;
  private final String name;
  private final String fullyQualifiedName;

  public AmbiguousDescriptor(String name, @Nullable String fullyQualifiedName, Set<Descriptor> descriptors) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.descriptors = descriptors;
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
      if (!descriptors.stream().map(Descriptor::fullyQualifiedName).allMatch(fqn -> Objects.equals(firstSymbol.fullyQualifiedName(), fqn))) {
        throw new IllegalArgumentException("Ambiguous symbol should contain descriptors with the same name");
      }
      // Here we have descriptors having same FQN but different local names, so we cannot assign any name to resulting value
      resultingSymbolName = "";
    }
    return new AmbiguousDescriptor(resultingSymbolName, firstSymbol.fullyQualifiedName(), flattenAmbiguousSymbols(descriptors));
  }

  private static Set<Descriptor> flattenAmbiguousSymbols(Set<Descriptor> descriptors) {
    Set<Descriptor> alternatives = new HashSet<>();
    for (Descriptor descriptor : descriptors) {
      if (descriptor.kind() == Kind.AMBIGUOUS) {
        Set<Descriptor> flattenedAlternatives = flattenAmbiguousSymbols(((AmbiguousDescriptor) descriptor).alternatives());
        alternatives.addAll(flattenedAlternatives);
      } else {
        alternatives.add(descriptor);
      }
    }
    return alternatives;
  }
}
