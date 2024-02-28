/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nonnull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.FileInputImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6903")
public class DateTimeUseTimeZoneAwareConstructors extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Using timezone aware \"datetime\"s should be preferred over using \"datetime.datetime.utcnow\" and \"datetime.datetime.utcfromtimestamp\"";
  private static final String UTCNOW_FQN = "datetime.datetime.utcnow";
  private static final String UTCFROMTIMESTAMP_FQN = "datetime.datetime.utcfromtimestamp";
  private static final Set<String> NON_COMPLIANT_FQNS = Set.of(UTCNOW_FQN, UTCFROMTIMESTAMP_FQN);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, DateTimeUseTimeZoneAwareConstructors::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();

    if (calleeSymbol != null) {
      String fullyQualifiedName = calleeSymbol.fullyQualifiedName();
      if (fullyQualifiedName == null || !NON_COMPLIANT_FQNS.contains(fullyQualifiedName)) {
        return;
      }
      var issue = context.addIssue(callExpression, MESSAGE);
      addQuickFix(context, issue, calleeSymbol);
    }
  }

  private static boolean isNameAlreadyUsed(SubscriptionContext context, String name) {
    Symbol foundSymbol = null;
    Tree tree = context.syntaxNode();
    while (foundSymbol == null && tree != null) {
      if (tree.is(Tree.Kind.FUNCDEF)) {
        foundSymbol = ((FunctionDef) tree).localVariables().stream().filter(symbol1 -> name.equals(symbol1.name())).findAny().orElse(null);
      } else if (tree.is(Tree.Kind.FILE_INPUT)) {
        foundSymbol = ((FileInputImpl) tree).globalVariables().stream().filter(symbol1 -> name.equals(symbol1.name())).findAny().orElse(null);
      }
      tree = TreeUtils.firstAncestor(tree, a -> a.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT));
    }
    return foundSymbol != null;
  }

  private static void addQuickFix(SubscriptionContext context, PreciseIssue issue, @Nonnull Symbol calleeSymbol) {
    if (!isFoundTimezoneImport(context) && !isNameAlreadyUsed(context, "timezone") && !isNameAlreadyUsed(context, "now")) {
      List<PythonTextEdit> pythonTextEdits = new ArrayList<>();
      CallExpression callExpression = (CallExpression) context.syntaxNode();
      String fullyQualifiedName = calleeSymbol.fullyQualifiedName();
      Expression calleeExpression = callExpression.callee();

      if (fullyQualifiedName == null || !calleeExpression.is(Tree.Kind.QUALIFIED_EXPR)) {
        return;
      }
      String quickFixDescription;
      if (UTCNOW_FQN.equals(fullyQualifiedName) && callExpression.arguments().isEmpty()) {
        pythonTextEdits.add(TextEditUtils.replace(((QualifiedExpression) calleeExpression).name(), "now"));
        pythonTextEdits.add(TextEditUtils.insertBefore(callExpression.rightPar(), "timezone.utc"));
        quickFixDescription = "utcnow";
      } else if (UTCFROMTIMESTAMP_FQN.equals(fullyQualifiedName) && callExpression.arguments().size() == 1) {
        pythonTextEdits.add(TextEditUtils.replace(((QualifiedExpression) calleeExpression).name(), "fromtimestamp"));
        pythonTextEdits.add(TextEditUtils.insertBefore(callExpression.rightPar(), ", timezone.utc"));
        quickFixDescription = "utcfromtimestamp";
      } else {
        return;
      }
      var quickFixImportBuilder = PythonQuickFix.newQuickFix(String.format("Change the %s call to construct a timezone aware datetime " +
        "instead", quickFixDescription));
      quickFixImportBuilder.addTextEdit(TextEditUtils.insertLineBefore(context.syntaxNode(), "from datetime import timezone"));
      quickFixImportBuilder.addTextEdit(pythonTextEdits);
      issue.addQuickFix(quickFixImportBuilder.build());
    }
  }

  private static boolean isFoundTimezoneImport(SubscriptionContext context) {
    var current = (StatementList) TreeUtils.firstAncestorOfKind(context.syntaxNode(), Tree.Kind.STATEMENT_LIST);
    boolean foundTimezoneImport = false;
    while (current != null && !foundTimezoneImport) {
      for (var a : current.statements()) {
        if (a.is(Tree.Kind.IMPORT_FROM)) {
          foundTimezoneImport = ((ImportFrom) a).importedNames().stream()
            .flatMap(importedName -> importedName.dottedName().names().stream())
            .map(HasSymbol::symbol)
            .filter(Objects::nonNull)
            .anyMatch(symbol -> "datetime.timezone".equals(symbol.fullyQualifiedName()));
        }
      }
      current = (StatementList) TreeUtils.firstAncestorOfKind(current, Tree.Kind.STATEMENT_LIST);
    }
    return foundTimezoneImport;
  }
}
