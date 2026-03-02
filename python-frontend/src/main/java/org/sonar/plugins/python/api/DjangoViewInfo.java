/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.api;

import java.util.HashSet;
import java.util.Set;

/**
 * Holds metadata about a Django view function discovered during project analysis.
 */
public record DjangoViewInfo(Set<String> urlPatterns) {

  public DjangoViewInfo {
    // Defensive copy to ensure immutability
    urlPatterns = Set.copyOf(urlPatterns);
  }

  public static DjangoViewInfo withoutPatterns() {
    return new DjangoViewInfo(Set.of());
  }

  public static DjangoViewInfo withPattern(String urlPattern) {
    return new DjangoViewInfo(Set.of(urlPattern));
  }

  public DjangoViewInfo addPattern(String urlPattern) {
    var newPatterns = new HashSet<>(urlPatterns);
    newPatterns.add(urlPattern);
    return new DjangoViewInfo(newPatterns);
  }
}
