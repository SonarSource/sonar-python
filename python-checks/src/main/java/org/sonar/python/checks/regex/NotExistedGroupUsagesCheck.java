/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks.regex;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
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
public class NotExistedGroupUsagesCheck extends AbstractRegexCheck {

  private static final Set<String> MATCH_CREATION_FUNCTION_NAMES = Set.of(
    "re.match",
    "re.fullmatch",
    "re.search"
  );
  private static final Set<String> COMPILE_FUNCTION_NAMES = Set.of("re.compile");


  public static final String GROUP_NAME_DOESNT_EXISTS_MESSAGE_FORMAT = "There is no group named '%s' in the regular expression.";
  public static final String GROUP_NUMBER_DOESNT_EXISTS_MESSAGE_FORMAT = "There is no group with number '%d' in the regular expression.";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    var groupsCollector = new KnownGroupsCollector();
    groupsCollector.visit(regexParseResult);

    getCallExpressionResultUsages(regexFunctionCall, COMPILE_FUNCTION_NAMES)
//      .filter(patternUsages -> patternUsages.size() == 2)
      .map(patternUsages -> getUsagesQualifiedExpressions(patternUsages, "typing.Pattern.match")
        .map(qe -> TreeUtils.firstAncestorOfKind(qe, Tree.Kind.ASSIGNMENT_STMT))
        .filter(Objects::nonNull)
        .map(AssignmentStatement.class::cast)
        .map(NotExistedGroupUsagesCheck::getAssignmentResultUsages)
        // To avoid FP apply rule only if there is only one match assignment
        .filter(NotExistedGroupUsagesCheck::isSingleAssignment)
        .flatMap(Collection::stream)
        .collect(Collectors.toList()))
      .or(() -> getCallExpressionResultUsages(regexFunctionCall, MATCH_CREATION_FUNCTION_NAMES))
      .ifPresent(matchUsages -> checkGroupAccesses(regexParseResult, groupsCollector, matchUsages));
  }

  private static Optional<List<Usage>> getCallExpressionResultUsages(CallExpression regexFunctionCall, Set<String> expressionFQNs) {
    return Optional.of(regexFunctionCall)
      .filter(c -> expressionFQNs.contains(TreeUtils.fullyQualifiedNameFromExpression(c)))
      .map(call -> TreeUtils.firstAncestorOfKind(call, Tree.Kind.ASSIGNMENT_STMT))
      .map(AssignmentStatement.class::cast)
      .map(NotExistedGroupUsagesCheck::getAssignmentResultUsages)
      .filter(NotExistedGroupUsagesCheck::isSingleAssignment);
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
    getUsagesQualifiedExpressions(matchUsages, "typing.Match.group")
      .map(qe -> TreeUtils.firstAncestorOfKind(qe, Tree.Kind.CALL_EXPR))
      .filter(Objects::nonNull)
      .map(CallExpression.class::cast)
      .map(CallExpression::arguments)
      .flatMap(Collection::stream)
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .forEach(argumentExpression -> checkGroupNameOrNumberExists(regexParseResult, groupsCollector, argumentExpression));
  }

  private void checkGroupNameOrNumberExists(RegexParseResult regexParseResult, KnownGroupsCollector groupsCollector, Expression argumentExpression) {
    Optional.of(argumentExpression)
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .filter(Predicate.not(groupsCollector.byName::containsKey))
      .map(NotExistedGroupUsagesCheck::getMessage)
      .or(() -> Optional.of(argumentExpression)
        .filter(NumericLiteral.class::isInstance)
        .map(NumericLiteral.class::cast)
        .map(NumericLiteral::valueAsLong)
        .filter(Predicate.not(groupsCollector.byNumber::containsKey))
        .map(NotExistedGroupUsagesCheck::getMessage))
      .ifPresent(message -> {
        IssueLocation issueLocation = PythonRegexIssueLocation.preciseLocation(regexParseResult.getResult(), message);
        regexContext.addIssue(argumentExpression, message)
          .secondary(issueLocation);
      });
  }

  private static String getMessage(String groupName) {
    return String.format(GROUP_NAME_DOESNT_EXISTS_MESSAGE_FORMAT, groupName);
  }

  private static String getMessage(Long groupNumber) {
    return String.format(GROUP_NUMBER_DOESNT_EXISTS_MESSAGE_FORMAT, groupNumber);
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
      .collect(Collectors.toList());
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
