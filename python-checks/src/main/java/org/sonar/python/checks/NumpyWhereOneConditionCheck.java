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

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S6729")
public class NumpyWhereOneConditionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"np.nonzero\" when only the condition parameter is provided to \"np.where\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyWhereOneConditionCheck::checkNumpyWhereCall);
  }

  private static void checkNumpyWhereCall(SubscriptionContext ctx) {
    CallExpression ce = (CallExpression) ctx.syntaxNode();
    Symbol symbol = ce.calleeSymbol();
    Optional.ofNullable(symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("numpy.where"::equals)
      .filter(fqn -> hasOneParameter(ce))
      .ifPresent(fqn -> {
        PreciseIssue issue = ctx.addIssue(ce, MESSAGE);
        addQuickFix(ce, issue);
      });
  }

  private static void addQuickFix(CallExpression ce, PreciseIssue issue) {
    Optional.of(ce.callee())
      .filter(exp -> exp.is(Tree.Kind.QUALIFIED_EXPR))
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::name)
      .map(NumpyWhereOneConditionCheck::getQuickFix)
      .ifPresent(issue::addQuickFix);
  }

  private static PythonQuickFix getQuickFix(Name qe) {
    return PythonQuickFix.newQuickFix("Replace numpy.where with numpy.nonzero")
      .addTextEdit((TextEditUtils.replace(qe, "nonzero")))
      .build();
  }

  private static boolean hasOneParameter(CallExpression ce) {
    List<Argument> argList = ce.arguments();
    if (argList.size() != 1 || argList.get(0).is(Tree.Kind.UNPACKING_EXPR)) {
      return false;
    }
    RegularArgument regArg = (RegularArgument) argList.get(0);
    Name keywordArgument = regArg.keywordArgument();
    if (keywordArgument == null) {
      return true;
    }
    return Optional.ofNullable(keywordArgument.name()).filter(name -> "condition".equals(keywordArgument.name())).isPresent();
  }
}
