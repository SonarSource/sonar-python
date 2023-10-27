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
package org.sonar.python.checks;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6796")
public class GenericFunctionTypeParameterCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a generic type parameter for this function instead of a \"TypeVar\".";
  private static final String SECONDARY_MESSAGE_USE = "Use of \"TypeVar\" here.";
  private static final String SECONDARY_MESSAGE_ASSIGNMENT = "\"TypeVar\" is assigned here.";
  public static final String TYPE_VAR_FQN = "typing.TypeVar";

  private static final Logger LOG = LoggerFactory.getLogger(GenericFunctionTypeParameterCheck.class);

  public static final String QUICKFIX_MESSAGE = "Replace use of generics in this function with type parameters";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, GenericFunctionTypeParameterCheck::checkUseOfGenerics);
  }

  private static void checkUseOfGenerics(SubscriptionContext ctx) {
    LOG.info("Starting the analysis of python file");
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    LOG.info("Starting the analysis of python 3.12 file");
    LOG.info(ctx.sourcePythonVersions().toString());
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    Set<Tree> usageLocations = Optional.ofNullable(functionDef.parameters())
      .map(ParameterList::nonTuple)
      .stream()
      .flatMap(List::stream)
      .map(Parameter::typeAnnotation)
      .filter(Objects::nonNull)
      .map(TypeAnnotation::expression)
      .filter(GenericFunctionTypeParameterCheck::isGenericTypeAnnotation)
      .collect(Collectors.toSet());
    Optional.ofNullable(functionDef.returnTypeAnnotation())
      .map(TypeAnnotation::expression)
      .filter(GenericFunctionTypeParameterCheck::isGenericTypeAnnotation)
      .ifPresent(usageLocations::add);

    if (!usageLocations.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(functionDef.name(), MESSAGE);
      usageLocations.forEach(loc -> issue.secondary(loc, SECONDARY_MESSAGE_USE));
      Set<Tree> assignmentLocations = getAssignmentLocations(usageLocations);
      assignmentLocations.forEach(loc -> issue.secondary(loc, SECONDARY_MESSAGE_ASSIGNMENT));
      addQuickFix(issue, functionDef, usageLocations, assignmentLocations);
    }
  }

  private static Set<Tree> getAssignmentLocations(Set<Tree> usageLocations) {
    return usageLocations.stream()
      .map(Name.class::cast)
      .map(Expressions::singleAssignedValue)
      .collect(Collectors.toSet());
  }

  private static boolean isGenericTypeAnnotation(Expression expression) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Expressions::singleAssignedValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::callee)
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter(TYPE_VAR_FQN::equals)
      .isPresent();
  }

  private static void addQuickFix(PreciseIssue issue, FunctionDef functionDef, Set<Tree> usageLocations, Set<Tree> assignmentLocations) {
    Set<AssignmentStatement> singleAssignmentLocations = assignmentLocations
      .stream()
      .map(GenericFunctionTypeParameterCheck::getSingleAssignmentStatements)
      .filter(Optional::isPresent)
      .map(Optional::get)
      .collect(Collectors.toSet());
    if (singleAssignmentLocations.isEmpty()) {
      return;
    }

    String replacementString = singleAssignmentLocations.stream()
      .map(GenericFunctionTypeParameterCheck::getTypeAnnotationString)
      .filter(Optional::isPresent)
      .map(Optional::get)
      .sorted(Comparator.naturalOrder())
      .collect(Collectors.joining(", "));


    List<PythonTextEdit> textEdits = singleAssignmentLocations.stream()
      .filter(GenericFunctionTypeParameterCheck::notUsedElsewhere)
      .map(assignmentStatement -> TextEditUtils
        .removeRange(
          assignmentStatement.firstToken().line(),
          assignmentStatement.firstToken().column(),
          assignmentStatement.lastToken().line(),
          assignmentStatement.lastToken().column() + 2))
      .collect(Collectors.toList());

    PythonTextEdit typeParameterTextEdit = Optional.of(functionDef)
      .map(FunctionDef::typeParams)
      .map(tp -> String.format(", %s", replacementString))
      .map(str -> TextEditUtils.insertBefore(functionDef.typeParams().rightBracket(), str))
      .orElse(TextEditUtils.insertAfter(functionDef.name(), String.format("[%s]", replacementString)));
    textEdits.add(typeParameterTextEdit);
    PythonQuickFix quickFix = PythonQuickFix
      .newQuickFix(QUICKFIX_MESSAGE)
      .addTextEdit(textEdits)
      .build();
    issue.addQuickFix(quickFix);
  }

  private static boolean notUsedElsewhere(AssignmentStatement assignmentStatement) {
    // TODO: implement to return true if the single assignment statement is used only in the context of the function analyzed.
    return true;
  }

  private static Optional<AssignmentStatement> getSingleAssignmentStatements(Tree tree) {
    return Optional.of(tree.parent())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(AssignmentStatement.class))
      .filter(assignmentStatement -> assignmentStatement.lhsExpressions().size() == 1)
      .filter(assignmentStatement -> assignmentStatement.lhsExpressions().get(0).expressions().size() == 1);
  }


  private static Optional<String> getTypeAnnotationString(AssignmentStatement assignmentStatement) {
    return getTypeVariableName(assignmentStatement)
      .map(Name::name)//Optional would be empty in the event that we have multiple assignments on the LHS.
      .map(name -> name.concat(getTypeAnnotationType(assignmentStatement)));
  }

  private static Optional<Name> getTypeVariableName(AssignmentStatement singleAssignmentStatement) {
    return Optional.of(singleAssignmentStatement)
      .map(AssignmentStatement::lhsExpressions)
      .map(list -> list.get(0))
      .map(ExpressionList::expressions)
      .map(list -> list.get(0))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class));
  }

  private static String getTypeAnnotationType(AssignmentStatement assignmentStatement) {
    return getTypeVariableBoundString(assignmentStatement.assignedValue())
      .or(() -> getTypeVariableConstraints(assignmentStatement.assignedValue()))
      .map(": "::concat)
      .orElse("");
  }

  private static Optional<String> getTypeVariableBoundString(Expression expression) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::arguments)
      .map(argList -> TreeUtils.argumentByKeyword("bound", argList))
      .map(RegularArgument::expression)
      .map(expr -> TreeUtils.treeToString(expr, false));
  }

  private static Optional<String> getTypeVariableConstraints(Expression expression) {
    //TODO: To be implemented when TypeVar is declared with type constraints.
//    Optional.of(expression)
//      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
//      .map(CallExpression::arguments)
//      .map(list -> list.remove(0))
    return Optional.empty();
  }


  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    PythonVersionUtils.Version required = PythonVersionUtils.Version.V_312;

    // All versions must be greater than or equal to the required version.
    return ctx.sourcePythonVersions().stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }
}
