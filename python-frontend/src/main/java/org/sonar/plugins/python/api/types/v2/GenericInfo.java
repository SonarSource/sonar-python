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
package org.sonar.plugins.python.api.types.v2;

import org.sonar.api.Beta;

/**
 * Declaration-level generic information, independent of supplied type arguments.
 * Unknown genericity is distinct from a known generic whose omitted arguments are unknown.
 */
@Beta
public sealed interface GenericInfo {
  GenericInfo UNKNOWN = new Unknown();
  GenericInfo NON_GENERIC = new NonGeneric();

  /** Compatibility query: false does not establish that a class is non-generic. */
  default boolean isGeneric() {
    return this instanceof Generic;
  }

  /** Legacy flags establish genericity only when true; they carry no default information. */
  static GenericInfo fromLegacy(boolean isGeneric) {
    return isGeneric ? new Generic(BareTypeArgumentStatus.UNKNOWN) : UNKNOWN;
  }

  record Unknown() implements GenericInfo {
  }

  record NonGeneric() implements GenericInfo {
  }

  /** Information about effective type arguments when no explicit arguments are supplied. */
  record Generic(BareTypeArgumentStatus bareTypeArguments) implements GenericInfo {
  }

  enum BareTypeArgumentStatus {
    /** Unsupported constructs, unresolved dependencies, or unavailable metadata. */
    UNKNOWN,
    /** At least one effective argument contains Any, including an omitted required parameter. */
    CONTAINS_ANY,
    /** Every effective argument has been established to be free of Any. */
    NO_ANY
  }
}
