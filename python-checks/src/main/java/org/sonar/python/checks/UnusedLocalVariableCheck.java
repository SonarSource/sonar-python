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

import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.ImportedNamesCollector;
import org.sonar.python.checks.utils.StringLiteralValuesCollector;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.FileInputImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1481")
public class UnusedLocalVariableCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT = "(_[a-zA-Z0-9_]*|dummy|unused|ignored)";
  private static final String MESSAGE = "Remove the unused local variable \"%s\".";
  private static final String SEQUENCE_UNPACKING_MESSAGE = "Replace the unused local variable \"%s\" with \"_\".";
  private static final String LOOP_INDEX_MESSAGE = "Replace the unused loop index \"%s\" with \"_\".";
  private static final String RENAME_QUICK_FIX_MESSAGE = "Replace with \"_\"";
  private static final String EXCEPT_CLAUSE_QUICK_FIX_MESSAGE = "Remove the unused local variable";
  private static final String ASSIGNMENT_QUICK_FIX_MESSAGE = "Remove assignment target";
  private static final String SECONDARY_MESSAGE = "Assignment to unused local variable \"%s\".";

  @RuleProperty(
    key = "regex",
    description = "Regular expression used to identify variable name to ignore.",
    defaultValue = DEFAULT)
  public String format = DEFAULT;
  private Pattern pattern;
  private boolean isTemplateVariablesAccessEnabled = false;

  @Override
  public void initialize(Context context) {
    pattern = Pattern.compile(format);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::checkTemplateVariablesAccessEnabled);
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkLocalVars(ctx, ctx.syntaxNode(),
      ((FunctionDef) ctx.syntaxNode()).localVariables()));
    context.registerSyntaxNodeConsumer(Kind.DICT_COMPREHENSION, ctx -> checkLocalVars(ctx, ctx.syntaxNode(),
      ((DictCompExpression) ctx.syntaxNode()).localVariables()));
    context.registerSyntaxNodeConsumer(Kind.LIST_COMPREHENSION, ctx -> checkLocalVars(ctx, ctx.syntaxNode(),
      ((ComprehensionExpression) ctx.syntaxNode()).localVariables()));
    context.registerSyntaxNodeConsumer(Kind.SET_COMPREHENSION, ctx -> checkLocalVars(ctx, ctx.syntaxNode(),
      ((ComprehensionExpression) ctx.syntaxNode()).localVariables()));
    context.registerSyntaxNodeConsumer(Kind.GENERATOR_EXPR, ctx -> checkLocalVars(ctx, ctx.syntaxNode(),
      ((ComprehensionExpression) ctx.syntaxNode()).localVariables()));
  }

  private void checkTemplateVariablesAccessEnabled(SubscriptionContext ctx) {
    var importedNamesCollector = new ImportedNamesCollector();
    importedNamesCollector.collect(ctx.syntaxNode());
    isTemplateVariablesAccessEnabled = importedNamesCollector.anyMatches("pandas"::equals);
  }

  private void checkLocalVars(SubscriptionContext ctx, Tree functionTree, Set<Symbol> symbols) {
    var stringLiteralValuesCollector = new StringLiteralValuesCollector();
    if (isTemplateVariablesAccessEnabled) {
      stringLiteralValuesCollector.collect(functionTree);
    }

    // https://docs.python.org/3/library/functions.html#locals
    if (CheckUtils.containsCallToLocalsFunction(functionTree)) {
      return;
    }
    symbols.stream()
      .filter(s -> !pattern.matcher(s.name()).matches())
      .filter(UnusedLocalVariableCheck::hasOnlyBindingUsages)
      .filter(UnusedLocalVariableCheck::isNotUpdatingParameterDict)
      .filter(symbol -> !isVariableAccessedInStringTemplate(symbol, stringLiteralValuesCollector))
      .forEach(symbol -> {
        var usages = symbol.usages().stream()
          .filter(usage -> usage.tree().parent() == null || !usage.tree().parent().is(Kind.PARAMETER))
          .filter(usage -> !isTupleDeclaration(usage))
          .filter(usage -> usage.kind() != Usage.Kind.FUNC_DECLARATION)
          .filter(usage -> usage.kind() != Usage.Kind.CLASS_DECLARATION)
          .toList();

        if (!usages.isEmpty()) {
          var firstUsage = usages.get(0);
          var issue = createIssue(ctx, symbol, firstUsage);

          usages.stream().skip(1)
            .forEach(usage -> issue.secondary(usage.tree(), String.format(SECONDARY_MESSAGE, symbol.name())));
        }
      });
  }

  private static boolean isVariableAccessedInStringTemplate(Symbol symbol, StringLiteralValuesCollector stringLiteralsCollector) {
    return stringLiteralsCollector.anyMatches(s -> s.matches(".*@" + symbol.name() + "((\\s+.*)|$)"));
  }

  public PreciseIssue createIssue(SubscriptionContext ctx, Symbol symbol, Usage usage) {
    if (isSequenceUnpacking(usage)) {
      var quickFix = PythonQuickFix.newQuickFix(RENAME_QUICK_FIX_MESSAGE, TextEditUtils.replace(usage.tree(), "_"));
      var issue = ctx.addIssue(usage.tree(), String.format(SEQUENCE_UNPACKING_MESSAGE, symbol.name()));
      issue.addQuickFix(quickFix);
      return issue;
    } else if (isLoopIndex(usage, symbol)) {
      PreciseIssue issue = ctx.addIssue(usage.tree(), String.format(LOOP_INDEX_MESSAGE, symbol.name()));
      if (isUnderscoreSymbolAlreadyAssigned(ctx, usage)) {
        PythonQuickFix quickFix = PythonQuickFix.newQuickFix(RENAME_QUICK_FIX_MESSAGE, TextEditUtils.replace(usage.tree(), "_"));
        issue.addQuickFix(quickFix);
      }
      return issue;
    } else {
      var issue = ctx.addIssue(usage.tree(), String.format(MESSAGE, symbol.name()));
      createExceptClauseQuickFix(usage, issue);
      createAssignmentQuickFix(usage, issue);
      return issue;
    }
  }

  private static void createAssignmentQuickFix(Usage usage, PreciseIssue issue) {
    if (usage.kind().equals(Usage.Kind.ASSIGNMENT_LHS)) {
      Statement assignmentStatement = ((Statement) TreeUtils.firstAncestorOfKind(usage.tree(), Kind.ASSIGNMENT_STMT,
        Kind.ANNOTATED_ASSIGNMENT));

      Optional.ofNullable(assignmentStatement).filter(stmt -> stmt.is(Kind.ASSIGNMENT_STMT)).map(AssignmentStatement.class::cast).ifPresent(stmt -> {
        PythonQuickFix quickFix = PythonQuickFix.newQuickFix(ASSIGNMENT_QUICK_FIX_MESSAGE,
          TextEditUtils.removeUntil(usage.tree(), stmt.assignedValue().firstToken()));
        issue.addQuickFix(quickFix);
      });

      Optional.ofNullable(assignmentStatement).filter(stmt -> stmt.is(Kind.ANNOTATED_ASSIGNMENT)).map(AnnotatedAssignment.class::cast)
        .map(AnnotatedAssignment::assignedValue).ifPresent(assignedValue -> {
          PythonQuickFix quickFix = PythonQuickFix.newQuickFix(ASSIGNMENT_QUICK_FIX_MESSAGE,
            TextEditUtils.removeUntil(usage.tree(), assignedValue.firstToken()));
          issue.addQuickFix(quickFix);
        });

      Tree assignmentTree = TreeUtils.firstAncestorOfKind(usage.tree(), Kind.ASSIGNMENT_EXPRESSION);
      Optional.ofNullable(assignmentTree).map(AssignmentExpression.class::cast).ifPresent(assignmentExpr -> {
        PythonQuickFix quickFix = PythonQuickFix.newQuickFix(ASSIGNMENT_QUICK_FIX_MESSAGE,
          createAssignmentExpressionQuickFix(usage, assignmentExpr));
        issue.addQuickFix(quickFix);
      });
    }
  }

  private static PythonTextEdit createAssignmentExpressionQuickFix(final Usage usage, final AssignmentExpression assignmentExpression) {
    var expression = assignmentExpression.expression();
    var parent = assignmentExpression.parent();
    if (parent.is(Kind.PARENTHESIZED) && expression instanceof Name nameExpr) {
      return TextEditUtils.replace(parent, nameExpr.name());
    } else {
      return TextEditUtils.removeUntil(usage.tree(), expression.firstToken());
    }
  }

  private static boolean isUnderscoreSymbolAlreadyAssigned(SubscriptionContext ctx, Usage usage) {
    Symbol foundUnderscoreSymbol = null;
    Tree searchTree = usage.kind().equals(Usage.Kind.LOOP_DECLARATION) ? ctx.syntaxNode() : null;
    while (foundUnderscoreSymbol == null && searchTree != null) {
      if (searchTree.is(Kind.FUNCDEF)) {
        foundUnderscoreSymbol =
          ((FunctionDef) searchTree).localVariables().stream().filter(symbol1 -> "_".equals(symbol1.name())).findAny().orElse(null);
      } else if (searchTree.is(Kind.FILE_INPUT)) {
        foundUnderscoreSymbol =
          ((FileInputImpl) searchTree).globalVariables().stream().filter(symbol1 -> "_".equals(symbol1.name())).findAny().orElse(null);
      }
      searchTree = TreeUtils.firstAncestor(searchTree, a -> a.is(Kind.FUNCDEF, Kind.FILE_INPUT));
    }
    return foundUnderscoreSymbol == null;
  }

  private static boolean isLoopIndex(Usage usage, Symbol symbol) {
    var allowedKinds = EnumSet.of(Usage.Kind.LOOP_DECLARATION, Usage.Kind.COMP_DECLARATION);
    Optional<Symbol> optionalSymbol =
      Optional.of(usage).filter(u -> allowedKinds.contains(u.kind())).map(Usage::tree).map(a -> ((Name) a).symbol());
    return optionalSymbol.map(value -> value.equals(symbol)).orElse(false);
  }

  private static void createExceptClauseQuickFix(Usage usage, PreciseIssue issue) {
    Optional.of(usage)
      .filter(u -> u.kind() == Usage.Kind.EXCEPTION_INSTANCE)
      .map(Usage::tree)
      .map(Tree::parent)
      .filter(ExceptClause.class::isInstance)
      .map(ExceptClause.class::cast)
      .filter(ec -> Objects.nonNull(ec.exception()))
      .map(ec -> {
        var replacement = TreeUtils.treeToString(ec.exception(), false) + ":";
        var from = ec.exception();
        var to = ec.colon();
        var textEdit = TextEditUtils.replaceRange(from, to, replacement);
        return PythonQuickFix.newQuickFix(EXCEPT_CLAUSE_QUICK_FIX_MESSAGE, textEdit);
      })
      .ifPresent(issue::addQuickFix);
  }

  private static boolean hasOnlyBindingUsages(Symbol symbol) {
    List<Usage> usages = symbol.usages();
    if (isOnlyTypeAnnotation(usages)) {
      return false;
    }
    return usages.stream().noneMatch(usage -> usage.kind() == Usage.Kind.IMPORT)
           && usages.stream().allMatch(Usage::isBindingUsage);
  }

  private static boolean isNotUpdatingParameterDict(Symbol symbol) {
    List<Usage> usages = symbol.usages();
    return usages.stream().noneMatch(UnusedLocalVariableCheck::isDictAssignmentExpressionUsage) ||
           usages.stream().noneMatch(usage -> usage.kind() == Usage.Kind.PARAMETER);
  }

  private static boolean isDictAssignmentExpressionUsage(Usage usage) {
    Tree compoundAssignmentTree = TreeUtils.firstAncestorOfKind(usage.tree(), Kind.COMPOUND_ASSIGNMENT);
    return compoundAssignmentTree instanceof CompoundAssignmentStatement compoundAssignmentStatement &&
           "|=".equals(compoundAssignmentStatement.compoundAssignmentToken().value()) &&
           compoundAssignmentStatement.lhsExpression().type().mustBeOrExtend("dict");
  }

  private static boolean isOnlyTypeAnnotation(List<Usage> usages) {
    return usages.size() == 1 && usages.get(0).isBindingUsage() &&
           TreeUtils.firstAncestor(usages.get(0).tree(),
             t -> t.is(Kind.ANNOTATED_ASSIGNMENT) && ((AnnotatedAssignment) t).assignedValue() == null) != null;
  }

  private static boolean isTupleDeclaration(Usage usage) {
    var tree = usage.tree();

    Predicate<Tree> isTupleDeclaration = t -> t.is(Kind.TUPLE)
                                              || (t.is(Kind.EXPRESSION_LIST) && ((ExpressionList) t).expressions().size() > 1)
                                              || (t.is(Kind.FOR_STMT)
                                                  && ((ForStatement) t).expressions().size() > 1
                                                  && tree instanceof Expression treeExpr
                                                  && ((ForStatement) t).expressions().contains(treeExpr));

    return !isSequenceUnpacking(usage) && TreeUtils.firstAncestor(tree, isTupleDeclaration) != null;
  }

  private static boolean isSequenceUnpacking(Usage usage) {
    return Optional.of(usage)
      .filter(u -> u.kind() == Usage.Kind.ASSIGNMENT_LHS)
      .map(Usage::tree)
      .map(tree -> TreeUtils.firstAncestorOfKind(tree, Kind.EXPRESSION_LIST))
      .map(ExpressionList.class::cast)
      .filter(list -> list.expressions().size() > 1)
      .isPresent();
  }
}
