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

import java.util.Comparator;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.CfgUtils;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis;
import org.sonar.python.checks.utils.DeadStoreUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.utils.DeadStoreUtils.isParameter;
import static org.sonar.python.checks.utils.DeadStoreUtils.isUsedInSubFunction;

@Rule(key = "S1226")
public class IgnoredParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Introduce a new variable or use its initial value before reassigning '%s'.";
  private static final String SECONDARY_MESSAGE_TEMPLATE = "'%s' is reassigned here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      ControlFlowGraph cfg = ctx.cfg(functionDef);
      if (cfg == null) {
        return;
      }
      LiveVariablesAnalysis lva = ctx.lva(functionDef);
      if (lva == null) {
        return;
      }

      Set<CfgBlock> unreachableBlocks = CfgUtils.unreachableBlocks(cfg);
      cfg.blocks().forEach(block -> {
        var unnecessaryAssignments = DeadStoreUtils.findUnnecessaryAssignments(block, lva.getLiveVariables(block), functionDef);
        unnecessaryAssignments.stream()
          .filter(assignment -> !"_".equals(assignment.symbol.name()))
          .filter((assignment -> isParameter(assignment.element)))
          // symbols should have at least two binding usages
          .filter(assignment -> assignment.symbol.usages().stream().filter(Usage::isBindingUsage).count() > 1)
          // no usages in unreachable blocks
          .filter(assignment -> !isSymbolUsedInUnreachableBlocks(lva, unreachableBlocks, assignment.symbol))
          .filter((assignment -> !isUsedInSubFunction(assignment.symbol, functionDef)))
          .forEach(assignment -> {
            var issue = ctx.addIssue(assignment.element, String.format(MESSAGE_TEMPLATE, assignment.symbol.name()));
            getQuickFix(functionDef, assignment.symbol).ifPresent(issue::addQuickFix);
            assignment.symbol.usages().stream()
              .filter(Usage::isBindingUsage)
              .filter(u -> u.kind() != Usage.Kind.PARAMETER)
              .map(Usage::tree)
              .collect(TreeUtils.groupAssignmentByParentStatementList())
              .values()
              .stream()
              .sorted(TreeUtils.getTreeByPositionComparator())
              .map(IgnoredParameterCheck::mapToParentAssignmentStatementOrExpression)
              .forEach(tree -> issue.secondary(tree, String.format(SECONDARY_MESSAGE_TEMPLATE, assignment.symbol.name())));
          });
      });
    });
  }

  private static Tree mapToParentAssignmentStatementOrExpression(Tree tree) {
    var assignment = TreeUtils.firstAncestor(tree, parent -> parent.is(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ASSIGNMENT_EXPRESSION));
    if (assignment != null) {
      return assignment;
    }
    return tree;
  }

  private static boolean isSymbolUsedInUnreachableBlocks(LiveVariablesAnalysis lva, Set<CfgBlock> unreachableBlocks, Symbol symbol) {
    return unreachableBlocks.stream().anyMatch(b -> lva.getLiveVariables(b).isSymbolUsedInBlock(symbol));
  }

  private static Optional<PythonQuickFix> getQuickFix(FunctionDef functionDef, Symbol symbol) {
    var sortedUsages = symbol.usages().stream()
      .sorted(Comparator.comparing((Usage usage) -> usage.tree().firstToken().line())
        .thenComparing(usage -> usage.tree().firstToken().column()))
      .toList();

    var firstReassignment = sortedUsages.stream()
      .filter(Usage::isBindingUsage)
      .filter(usage -> usage.kind() != Usage.Kind.PARAMETER)
      .findFirst();

    if (firstReassignment.isEmpty() || !isSupportedQuickFixUsage(firstReassignment.get())) {
      return Optional.empty();
    }

    String newName = freshVariableName(functionDef, symbol.name());
    var edits = sortedUsages.stream()
      .filter(usage -> usage.kind() != Usage.Kind.PARAMETER)
      .filter(usage -> isSameOrAfter(usage.tree(), firstReassignment.get().tree()))
      .map(usage -> TextEditUtils.replace(usage.tree(), newName))
      .toList();

    if (edits.isEmpty()) {
      return Optional.empty();
    }

    return Optional.of(PythonQuickFix.newQuickFix(String.format("Rename the reassigned value to '%s'", newName))
      .addTextEdit(edits)
      .build());
  }

  private static boolean isSupportedQuickFixUsage(Usage usage) {
    return switch (usage.kind()) {
      case ASSIGNMENT_LHS, LOOP_DECLARATION, CLASS_DECLARATION -> true;
      default -> usage.tree().parent().is(Tree.Kind.ASSIGNMENT_EXPRESSION);
    };
  }

  private static boolean isSameOrAfter(Tree tree, Tree reference) {
    int line = tree.firstToken().line();
    int referenceLine = reference.firstToken().line();
    if (line != referenceLine) {
      return line > referenceLine;
    }
    return tree.firstToken().column() >= reference.firstToken().column();
  }

  private static String freshVariableName(FunctionDef functionDef, String originalName) {
    String baseName = originalName + "_value";
    Set<String> localNames = functionDef.localVariables().stream()
      .map(Symbol::name)
      .collect(java.util.stream.Collectors.toSet());

    String candidate = baseName;
    int suffix = 1;
    while (localNames.contains(candidate)) {
      candidate = baseName + "_" + suffix;
      suffix++;
    }
    return candidate;
  }
}
