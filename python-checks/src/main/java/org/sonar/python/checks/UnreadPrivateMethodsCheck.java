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

import java.util.Collection;
import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.semantic.BuiltinSymbols;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;

@Rule(key = "S1144")
public class UnreadPrivateMethodsCheck extends AbstractUnreadPrivateMembersCheck {
  @Override
  String memberPrefix() {
    return "__";
  }

  @Override
  Symbol.Kind kind() {
    return FUNCTION;
  }

  @Override
  String message(String memberName) {
    return "Remove this unused class-private '" + memberName + "' method.";
  }

  @Override
  String secondaryMessage() {
    return null;
  }

  @Override
  protected boolean isException(Symbol symbol) {
    return Optional.of(symbol)
      .filter(FunctionSymbol.class::isInstance)
      .map(FunctionSymbol.class::cast)
      .map(FunctionSymbol::decorators)
      .stream()
      .flatMap(Collection::stream)
      .anyMatch(Predicate.not(BuiltinSymbols.STATIC_AND_CLASS_METHOD_DECORATORS::contains));
  }

  @Override
  protected boolean hasAmbiguousUsage(Symbol symbol, ClassDef classDef) {
    CallNamesVisitor visitor = new CallNamesVisitor(symbol.name());
    classDef.accept(visitor);

    // There is always at least one usage - the declaration itself.
    return visitor.usages > 1;
  }

  private static class CallNamesVisitor extends BaseTreeVisitor {
    private final String name;
    private int usages = 0;

    public CallNamesVisitor(String name) {
      this.name = name;
    }

    @Override
    public void visitName(Name pyNameTree) {
      if (name.equals(pyNameTree.name())) {
        ++usages;
      }
    }
  }
}
