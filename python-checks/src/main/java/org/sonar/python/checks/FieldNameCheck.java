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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.python.checks.utils.CheckUtils;

@Rule(key = "S116")
public class FieldNameCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename this field \"%s\" to match the regular expression %s.";

  private static final String CONSTANT_PATTERN = "^[_A-Z][A-Z0-9_]*$";

  private static final String DEFAULT = "^[_a-z][_a-z0-9]*$";

  @RuleProperty(
    key = "format",
    description = "Regular expression used to check the field names against.",
    defaultValue = DEFAULT)
  public String format = DEFAULT;


  @Override
  public void initialize(Context context) {
    Pattern pattern = Pattern.compile(format);
    Pattern constantPattern = Pattern.compile(CONSTANT_PATTERN);
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      if (CheckUtils.classHasInheritance(classDef)) {
        return;
      }
      for (Symbol field : fieldsToCheck(classDef)) {
        String name = field.name();
        if (!pattern.matcher(name).matches() && !constantPattern.matcher(name).matches()) {
          String message = String.format(MESSAGE, name, this.format);
          field.usages().stream()
            .filter(usage -> usage.kind() == Usage.Kind.ASSIGNMENT_LHS)
            .limit(1)
            .forEach(usage -> ctx.addIssue(usage.tree(), message));
        }
      }
    });
  }

  private static List<Symbol> fieldsToCheck(ClassDef classDef) {
    Set<String> classFieldNames = classDef.classFields().stream().map(Symbol::name).collect(Collectors.toSet());
    List<Symbol> result = new ArrayList<>(classDef.classFields());
    classDef.instanceFields().stream().filter(f -> !classFieldNames.contains(f.name())).forEach(result::add);
    return result;
  }

}
