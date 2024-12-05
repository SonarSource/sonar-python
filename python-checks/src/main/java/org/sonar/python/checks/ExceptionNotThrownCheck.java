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

import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.BuiltinSymbols;

// https://jira.sonarsource.com/browse/RSPEC-3984
@Rule(key = "S3984")
public class ExceptionNotThrownCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Raise this exception or remove this useless statement.";
  public static final String QUICK_FIX_MESSAGE = "Raise this exception";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, check(ExceptionNotThrownCheck::symbolFromInvocation));
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, check(ExceptionNotThrownCheck::symbolFromName));
  }

  private static Consumer<SubscriptionContext> check(Function<Tree, Symbol> extractClassSymbol) {
    return subscriptionContext -> {
      Tree t = subscriptionContext.syntaxNode();
      Symbol symb = extractClassSymbol.apply(t);
      if (symb != null && symb.is(Symbol.Kind.CLASS) && isThrowable((ClassSymbol) symb)) {
        Tree parent = t.parent();
        if (parent.is(Tree.Kind.EXPRESSION_STMT)) {
          var issue = subscriptionContext.addIssue(t, MESSAGE);

          var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
            .addTextEdit(TextEditUtils.insertBefore(t, "raise "))
            .build();

          issue.addQuickFix(quickFix);
        }
      }
    };
  }

  private static final Set<String> BUILTIN_EXCEPTIONS_FQNS = Stream.of(
    BuiltinSymbols.EXCEPTIONS,
    BuiltinSymbols.EXCEPTIONS_PYTHON2).flatMap(Set::stream).filter(n -> !"WindowsError".equals(n)).collect(Collectors.toSet());

  private static boolean isThrowable(ClassSymbol cs) {
    return (BUILTIN_EXCEPTIONS_FQNS.contains(cs.fullyQualifiedName()) ||
      cs.superClasses().stream().map(Symbol::fullyQualifiedName).anyMatch(BUILTIN_EXCEPTIONS_FQNS::contains)) &&
      !"WindowsError".equals(cs.fullyQualifiedName());
  }

  @Nullable
  private static Symbol symbolFromInvocation(Tree t) {
    return ((CallExpression) t).calleeSymbol();
  }

  @Nullable
  private static Symbol symbolFromName(Tree t) {
    return ((Name) t).symbol();
  }

}
