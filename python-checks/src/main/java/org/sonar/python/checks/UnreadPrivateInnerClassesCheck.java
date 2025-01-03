/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;

@Rule(key = "S3985")
public class UnreadPrivateInnerClassesCheck extends AbstractUnreadPrivateMembersCheck {
  @Override
  String memberPrefix() {
    return "_";
  }

  @Override
  protected boolean isException(Symbol symbol) {
    return Optional.of(symbol)
      .filter(ClassSymbol.class::isInstance)
      .map(ClassSymbol.class::cast)
      .filter(ClassSymbol::hasDecorators)
      .isPresent();
  }

  @Override
  Symbol.Kind kind() {
    return CLASS;
  }

  @Override
  String message(String memberName) {
    return "Remove this unused private '" + memberName + "' class.";
  }

  @Override
  String secondaryMessage() {
    return null;
  }
}
