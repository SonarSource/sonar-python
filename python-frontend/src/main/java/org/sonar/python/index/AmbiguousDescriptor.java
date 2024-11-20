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

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

public class AmbiguousDescriptor implements Descriptor {

  private final Set<Descriptor> descriptors;
  private final String name;
  private final String fullyQualifiedName;

  public AmbiguousDescriptor(String name, @Nullable String fullyQualifiedName, Set<Descriptor> descriptors) {
    if (descriptors.stream().anyMatch(d -> d.kind().equals(Kind.AMBIGUOUS))) {
      throw new IllegalArgumentException("Ambiguous descriptors should not be nested.");
    }
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
}
