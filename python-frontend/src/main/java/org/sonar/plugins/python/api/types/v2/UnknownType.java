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

import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.TriBool;

@Beta
public sealed interface UnknownType extends PythonType permits UnknownTypeImpl, UnknownType.UnresolvedImportType {

  @Override
  default TriBool isCompatibleWith(PythonType another) {
    return TriBool.UNKNOWN;
  }

  record UnresolvedImportType(String importPath) implements UnknownType {
    @Override
    public Optional<PythonType> resolveMember(String memberName) {
      var memberFqn = Stream.of(importPath, memberName)
        .filter(Predicate.not(String::isEmpty))
        .collect(Collectors.joining("."));
      return Optional.of(new UnresolvedImportType(memberFqn));
    }
  }
}
