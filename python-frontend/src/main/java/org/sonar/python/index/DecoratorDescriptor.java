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
import java.util.Optional;
import javax.annotation.Nullable;

/** Flat type references for one decorator; no callable signatures or class member graphs. */
public record DecoratorDescriptor(List<TypeReference> expressionTypes, Optional<List<TypeReference>> calleeTypes) {
  public DecoratorDescriptor {
    expressionTypes = nonEmptyCopy(expressionTypes);
    calleeTypes = calleeTypes.map(DecoratorDescriptor::nonEmptyCopy);
  }

  private static List<TypeReference> nonEmptyCopy(List<TypeReference> types) {
    return types.isEmpty() ? List.of(TypeReference.UNKNOWN) : List.copyOf(types);
  }

  /**
   * A null name is an explicit unknown alternative. All names, including unresolved import paths, resolve through V2's lazy type context.
   * A name alone does not establish callable identity.
   */
  public record TypeReference(@Nullable String fullyQualifiedName, boolean isInstance) {
    public static final TypeReference UNKNOWN = new TypeReference(null, false);
  }
}
