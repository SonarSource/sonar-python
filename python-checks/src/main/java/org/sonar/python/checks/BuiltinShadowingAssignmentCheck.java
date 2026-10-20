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
package org.sonar.python.checks;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5806")
public class BuiltinShadowingAssignmentCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Rename this variable; it shadows a builtin.";
  public static final String REPEATED_VAR_MESSAGE = "Variable also assigned here.";
  public static final String RENAME_PREFIX = "_";
  public static final String QUICK_FIX_MESSAGE_FORMAT = "Rename to " + RENAME_PREFIX+ " %s";
  private final Map<SymbolV2, PreciseIssue> variableIssuesRaised = new HashMap<>();
  private static final Set<String> notBuiltins = Set.of("ellipsis", "function");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> variableIssuesRaised.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, this::checkAssignmentExpression);
  }

  private void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    Name lhsName = assignmentExpression.lhsName();
    if (shouldReportIssue(lhsName, ctx)) {
      raiseIssueForNonGlobalVariable(ctx, lhsName);
    }
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      for (int i = 0; i < assignment.lhsExpressions().size(); i++) {
        for (Expression expression : assignment.lhsExpressions().get(i).expressions()) {
          if (shouldReportIssue(expression, ctx)) {
            raiseIssueForNonGlobalVariable(ctx, (Name) expression);
          }
        }
      }
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext ctx) {
    AnnotatedAssignment assignment = (AnnotatedAssignment) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      Expression variable = assignment.variable();
      Token equalToken = assignment.equalToken();
      if (equalToken != null && shouldReportIssue(variable, ctx)) {
        raiseIssueForNonGlobalVariable(ctx, (Name) variable);
      }
    }
  }

  private static boolean hasRenameCollision(SymbolV2 symbol) {
    String replacement = RENAME_PREFIX + symbol.name();
    // Include referenced names and nested scopes: renaming a closure reference must not capture another binding.
    return symbol.usages().stream()
      .map(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.FUNCDEF))
      .filter(Objects::nonNull)
      .distinct()
      .anyMatch(function -> TreeUtils.hasDescendant(function, tree -> tree instanceof Name name && replacement.equals(name.name())));
  }

  private void raiseIssueForNonGlobalVariable(SubscriptionContext ctx, Name variable) {
    Tree scope = TreeUtils.firstAncestorOfKind(variable, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA);
    Optional.ofNullable(variable.symbolV2())
      .filter(symbol -> symbol.usages().stream()
        .filter(usage -> usage.kind() == UsageV2.Kind.GLOBAL_DECLARATION)
        .noneMatch(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.LAMBDA) == scope))
      .ifPresent(symbol -> {
        var existingIssue = variableIssuesRaised.get(symbol);
        if (existingIssue != null) {
          existingIssue.secondary(variable, REPEATED_VAR_MESSAGE);
        } else {
          var issue = ctx.addIssue(variable, MESSAGE);
          variableIssuesRaised.put(symbol, issue);

          // V2 can associate a nested global declaration with this local symbol.
          // Keep the local finding, but do not rename the conflated global usages.
          boolean hasGlobalUsage = symbol.usages().stream().anyMatch(usage -> usage.kind() == UsageV2.Kind.GLOBAL_DECLARATION);
          if (!hasGlobalUsage && !hasRenameCollision(symbol) && !hasComprehensionWalrus(symbol)) {
            var quickFix = createQuickFix(symbol);
            issue.addQuickFix(quickFix);
          }
        }
      });
  }

  private static boolean hasComprehensionWalrus(SymbolV2 symbol) {
    // V2 splits comprehension walrus targets from enclosing-scope references. Inspect
    // the function as well as the symbol so neither side can offer an incomplete rename.
    return symbol.usages().stream()
      .map(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.FUNCDEF))
      .filter(Objects::nonNull)
      .distinct()
      .anyMatch(function -> TreeUtils.hasDescendant(function, tree -> tree instanceof AssignmentExpression assignment
        && symbol.name().equals(assignment.lhsName().name()) && isInComprehension(assignment)));
  }

  private static boolean isInComprehension(AssignmentExpression assignment) {
    var scope = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.LIST_COMPREHENSION, Tree.Kind.SET_COMPREHENSION,
      Tree.Kind.DICT_COMPREHENSION, Tree.Kind.GENERATOR_EXPR, Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA);
    return scope != null && !scope.is(Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA);
  }

  private static PythonQuickFix createQuickFix(SymbolV2 symbol) {
    var edits = symbol.usages()
      .stream()
      .map(UsageV2::tree)
      .map(Tree::firstToken)
      .map(token -> TextEditUtils.insertBefore(token, RENAME_PREFIX))
      .toList();

    return PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE_FORMAT, symbol.name()))
      .addTextEdit(edits)
      .build();
  }

  private static boolean shouldReportIssue(Tree tree, SubscriptionContext ctx) {
    return tree.is(Tree.Kind.NAME) && isBuiltInName((Name) tree, ctx) && TreeUtils.firstAncestorOfKind(tree.parent(), Tree.Kind.FUNCDEF) != null;
  }

  private static boolean isBuiltInName(Name name, SubscriptionContext ctx) {
    if (notBuiltins.contains(name.name())) {
      return false;
    }
    return ctx.typeTable().getBuiltinsModule() instanceof ModuleType builtins && builtins.members().containsKey(name.name());
  }
}
