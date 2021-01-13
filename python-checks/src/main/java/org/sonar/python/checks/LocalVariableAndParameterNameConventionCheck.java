/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks;

import java.util.Comparator;
import java.util.EnumSet;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;

@Rule(key = "S117")
public class LocalVariableAndParameterNameConventionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename this %s \"%s\" to match the regular expression %s.";
  private static final String PARAMETER = "parameter";
  private static final String LOCAL_VAR = "local variable";
  private static final EnumSet<Usage.Kind> USAGES = EnumSet.of(Usage.Kind.PARAMETER, Usage.Kind.LOOP_DECLARATION, Usage.Kind.ASSIGNMENT_LHS);

  private static final String CONSTANT_PATTERN = "^[_A-Z][A-Z0-9_]*$";

  private static final String DEFAULT = "^[_a-z][a-z0-9_]*$";
  @RuleProperty(
    key = "format",
    description = "Regular expression used to check the names against.",
    defaultValue = DEFAULT)
  public String format = DEFAULT;
  private Pattern constantPattern;
  private Pattern pattern;


  @Override
  public void initialize(Context context) {
    pattern = Pattern.compile(format);
    constantPattern = Pattern.compile(CONSTANT_PATTERN);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef funcDef = (FunctionDef) ctx.syntaxNode();
      funcDef.localVariables().stream().sorted(Comparator.comparing(Symbol::name)).forEach(s -> checkName(s, ctx));
    });
  }

  private void checkName(Symbol symbol, SubscriptionContext ctx) {
    String name = symbol.name();
    if (!pattern.matcher(name).matches()) {
      symbol.usages().stream()
        .filter(usage -> USAGES.contains(usage.kind()))
        .sorted(Comparator.comparingInt(u -> u.tree().firstToken().line()))
        .limit(1)
        .forEach(usage -> raiseIssueForNameAndUsage(ctx, name, usage));
    }
  }

  private void raiseIssueForNameAndUsage(SubscriptionContext ctx, String name, Usage usage) {
    String type = PARAMETER;
    Usage.Kind kind = usage.kind();
    if (kind == Usage.Kind.ASSIGNMENT_LHS) {
      type = LOCAL_VAR;
      if (constantPattern.matcher(name).matches()) {
        return;
      }
    } else if (kind == Usage.Kind.LOOP_DECLARATION) {
      type = LOCAL_VAR;
      if (name.length() <= 1) {
        return;
      }
    }
    ctx.addIssue(usage.tree(), String.format(MESSAGE, type, name, format));
  }
}
