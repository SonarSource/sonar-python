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

import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.python.checks.utils.CheckUtils;

@Rule(key = "S1700")
public class FieldDuplicatesClassNameCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename field \"%s\"";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      if (CheckUtils.classHasInheritance(classDef)) {
        return;
      }
      String className = classDef.name().name();
      Set<Symbol> allFields = new HashSet<>(classDef.classFields());
      allFields.addAll(classDef.instanceFields());
      allFields.stream()
        .filter(symbol -> className.equalsIgnoreCase(symbol.name()))
        .filter(symbol -> symbol.usages().stream().noneMatch(usage -> usage.kind() == Usage.Kind.FUNC_DECLARATION))
        .forEach(symbol -> symbol.usages()
          .stream()
          .findFirst()
          .ifPresent(usage -> ctx.addIssue(usage.tree(), String.format(MESSAGE, symbol.name())).secondary(classDef.name(), "Class declaration")));
    });
  }
}
