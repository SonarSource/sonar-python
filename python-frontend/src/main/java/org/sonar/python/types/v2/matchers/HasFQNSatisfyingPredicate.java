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
package org.sonar.python.types.v2.matchers;

import java.util.Optional;
import java.util.function.Function;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.PythonType;

/**
 * Matches types whose FQN satisfies the given predicate.
 * Prefer convenience factories such as {@code TypeMatchers.withFQNPrefix} when applicable.
 */
public class HasFQNSatisfyingPredicate implements TypePredicate {

  private final Function<String, TriBool> fqnPredicate;

  public HasFQNSatisfyingPredicate(Function<String, TriBool> fqnPredicate) {
    this.fqnPredicate = fqnPredicate;
  }

  @Override
  public TriBool check(PythonType type, TypePredicateContext ctx) {
    return Optional.of(type)
      .flatMap(FullyQualifiedNameHelper::getFullyQualifiedName)
      .map(fqnPredicate)
      .orElse(TriBool.UNKNOWN);
  }
}
