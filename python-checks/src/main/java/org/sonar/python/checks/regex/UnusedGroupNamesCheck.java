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
package org.sonar.python.checks.regex;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.CapturingGroupTree;
import org.sonarsource.analyzer.commons.regex.ast.RegexBaseVisitor;

@Rule(key = "S5860")
public class UnusedGroupNamesCheck extends AbstractRegexCheck {

  private static final Set<String> MATCH_CREATION_FUNCTION_NAMES = Set.of(
    "re.match",
    "re.fullmatch",
    "re.search"
  );
  private static final Set<String> COMPILE_FUNCTION_NAMES = Set.of("re.compile");


  private static final String GROUP_NAME_DOESNT_EXISTS_MESSAGE_FORMAT = "There is no group named '%s' in the regular expression.";
  private static final String USE_NAME_INSTEAD_OF_NUMBER_MESSAGE_FORMAT = "Directly use '%s' instead of its group number.";
  private static final String GROUP_NAME_SECONDARY_MESSAGE_FORMAT = "Named group '%s'";
  private static final String GROUP_NUMBER_SECONDARY_MESSAGE_FORMAT = "Group %d";
  private static final String NO_GROUP_NAMES_SECONDARY_MESSAGE = "No named groups defined in this regular expression.";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    var groupsCollector = new KnownGroupsCollector();
    groupsCollector.visit(regexParseResult);

    getCallExpressionResultUsages(regexFunctionCall, COMPILE_FUNCTION_NAMES)
      .map(patternUsages -> getUsagesQualifiedExpressions(patternUsages, "re.Pattern.match")
        .map(qe -> TreeUtils.firstAncestorOfKind(qe, Tree.Kind.ASSIGNMENT_STMT))
        .filter(Objects::nonNull)
        .map(AssignmentStatement.class::cast)
        .map(UnusedGroupNamesCheck::getAssignmentResultUsages)
        // To avoid FP apply rule only if there is only one match assignment
        .filter(UnusedGroupNamesCheck::isSingleAssignment)
        .flatMap(Collection::stream)
        .toList())
      .or(() -> getCallExpressionResultUsages(regexFunctionCall, MATCH_CREATION_FUNCTION_NAMES))
      .ifPresent(matchUsages -> checkGroupAccesses(regexParseResult, groupsCollector, matchUsages));
  }

  private static Optional<List<Usage>> getCallExpressionResultUsages(CallExpression regexFunctionCall, Set<String> expressionFQNs) {
    return Optional.of(regexFunctionCall)
      .filter(c -> TreeUtils.fullyQualifiedNameFromExpression(c).filter(expressionFQNs::contains).isPresent())
      .map(call -> TreeUtils.firstAncestorOfKind(call, Tree.Kind.ASSIGNMENT_STMT))
      .map(AssignmentStatement.class::cast)
      .map(UnusedGroupNamesCheck::getAssignmentResultUsages)
      .filter(UnusedGroupNamesCheck::isSingleAssignment);
  }

  private static boolean isSingleAssignment(List<Usage> usages) {
    return getAssignmentsCount(usages) == 1;
  }

  private static long getAssignmentsCount(List<Usage> usages) {
    return usages.stream()
      .map(Usage::kind)
      .filter(kind -> kind == Usage.Kind.ASSIGNMENT_LHS)
      .count();
  }

  private void checkGroupAccesses(RegexParseResult regexParseResult, KnownGroupsCollector groupsCollector, List<Usage> matchUsages) {
    getUsagesQualifiedExpressions(matchUsages, "re.Match.group")
      .map(qe -> TreeUtils.firstAncestorOfKind(qe, Tree.Kind.CALL_EXPR))
      .filter(Objects::nonNull)
      .map(CallExpression.class::cast)
      .map(CallExpression::arguments)
      .flatMap(Collection::stream)
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .forEach(argumentExpression -> checkValidGroupNameAccess(regexParseResult, groupsCollector, argumentExpression));
  }

  private void checkValidGroupNameAccess(RegexParseResult regexParseResult, KnownGroupsCollector groupsCollector, Expression argumentExpression) {
    //Check if group accessed by name exists
    Optional.of(argumentExpression)
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .filter(Predicate.not(groupsCollector.byName::containsKey))
      .ifPresent(nonExistingGroupName -> {
        var message = getGroupNameNotExistsMessage(nonExistingGroupName);
        var issue = regexContext.addIssue(argumentExpression, message);

        if (groupsCollector.byName.isEmpty()) {
          var secondaryLocation = PythonRegexIssueLocation.preciseLocation(regexParseResult.getResult(), NO_GROUP_NAMES_SECONDARY_MESSAGE);
          issue.secondary(secondaryLocation);
        } else {
          groupsCollector.byName.forEach((groupName, group) -> {
            var secondaryMessage = String.format(GROUP_NAME_SECONDARY_MESSAGE_FORMAT, groupName);
            var secondaryLocation = PythonRegexIssueLocation.preciseLocation(group, secondaryMessage);
            issue.secondary(secondaryLocation);
          });
        }
      });

    //Check if group access by number doesn't have name
    Optional.of(argumentExpression)
      .filter(NumericLiteral.class::isInstance)
      .map(NumericLiteral.class::cast)
      .map(NumericLiteral::valueAsLong)
      .filter(groupsCollector.byNumber::containsKey)
      .map(groupsCollector.byNumber::get)
      .ifPresent(group -> {
        var message = getUseNameInsteadNumberMessage(group);
        var issue = regexContext.addIssue(argumentExpression, message);
        var secondaryMessage = String.format(GROUP_NUMBER_SECONDARY_MESSAGE_FORMAT, group.getGroupNumber());
        var secondaryLocation = PythonRegexIssueLocation.preciseLocation(group, secondaryMessage);
        issue.secondary(secondaryLocation);
      });
  }

  private static String getUseNameInsteadNumberMessage(CapturingGroupTree capturingGroupTree) {
    //it will never be null actually
    var name = capturingGroupTree.getName().orElse(null);
    return String.format(USE_NAME_INSTEAD_OF_NUMBER_MESSAGE_FORMAT, name);
  }

  private static String getGroupNameNotExistsMessage(String groupName) {
    return String.format(GROUP_NAME_DOESNT_EXISTS_MESSAGE_FORMAT, groupName);
  }

  private static Stream<QualifiedExpression> getUsagesQualifiedExpressions(List<Usage> usages, String fullyQualifiedName) {
    return usages.stream()
      .filter(usage -> usage.kind() == Usage.Kind.OTHER)
      .map(Usage::tree)
      .map(tree -> TreeUtils.firstAncestorOfKind(tree, Tree.Kind.QUALIFIED_EXPR))
      .filter(Objects::nonNull)
      .map(QualifiedExpression.class::cast)
      .filter(qe -> Optional.of(qe.name())
        .map(HasSymbol::symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(fullyQualifiedName::equals).isPresent());
  }

  private static List<Usage> getAssignmentResultUsages(AssignmentStatement assignment) {
    return assignment.lhsExpressions()
      .stream()
      .map(v -> TreeUtils.firstChild(v, Name.class::isInstance).map(Name.class::cast))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .map(HasSymbol::symbol)
      .filter(Objects::nonNull)
      .map(Symbol::usages)
      .flatMap(Collection::stream)
      .toList();
  }

  private static class KnownGroupsCollector extends RegexBaseVisitor {

    final Map<String, CapturingGroupTree> byName = new HashMap<>();
    final Map<Long, CapturingGroupTree> byNumber = new HashMap<>();

    @Override
    public void visitCapturingGroup(CapturingGroupTree tree) {
      tree.getName().ifPresent(name -> {
        byName.put(name, tree);
        byNumber.put((long) tree.getGroupNumber(), tree);
      });
      super.visitCapturingGroup(tree);
    }
  }
}
