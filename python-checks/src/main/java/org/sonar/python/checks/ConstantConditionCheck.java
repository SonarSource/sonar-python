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

import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.AND;
import static org.sonar.plugins.python.api.tree.Tree.Kind.GENERATOR_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.LAMBDA;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NONE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NOT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NUMERIC_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.OR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.QUALIFIED_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.UNPACKING_EXPR;

@Rule(key = "S5797")
public class ConstantConditionCheck extends PythonVisitorCheck {

  private static final String MESSAGE = "Replace this expression; used as a condition it will always be constant.";
  private static final List<String> ACCEPTED_DECORATORS = Arrays.asList("overload", "staticmethod", "classmethod");
  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void visitFileInput(FileInput fileInput) {
    reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(getContext().pythonFile());
    super.visitFileInput(fileInput);
  }

  @Override
  public void visitIfStatement(IfStatement ifStatement) {
    checkConstantCondition(ifStatement.condition());
    scan(ifStatement.body());
    scan(ifStatement.elifBranches());
    scan(ifStatement.elseBranch());
  }

  @Override
  public void visitConditionalExpression(ConditionalExpression conditionalExpression) {
    checkConstantCondition(conditionalExpression.condition());
    super.visitConditionalExpression(conditionalExpression);
  }

  @Override
  public void visitComprehensionIf(ComprehensionIf comprehensionIf) {
    checkConstantCondition(comprehensionIf.condition());
    super.visitComprehensionIf(comprehensionIf);
  }

  private void checkConstantCondition(Expression condition) {
    Expression constantBooleanExpression = getConstantBooleanExpression(condition);
    if (constantBooleanExpression != null) {
      addIssue(constantBooleanExpression, MESSAGE);
    }
    checkExpression(condition);
  }

  private static boolean isConstant(Expression condition) {
    return isImmutableConstant(condition) || isConstantCollectionLiteral(condition);
  }

  private static boolean isImmutableConstant(Expression condition) {
    return TreeUtils.isBooleanLiteral(condition) ||
      condition.is(NUMERIC_LITERAL, STRING_LITERAL, NONE, LAMBDA, GENERATOR_EXPR);
  }

  private static Expression getConstantBooleanExpression(Expression condition) {
    if (condition.is(AND, OR)) {
      BinaryExpression binaryExpression = (BinaryExpression) condition;
      if (isConstant(binaryExpression.leftOperand())) {
        return binaryExpression.leftOperand();
      }
      if (isConstant(binaryExpression.rightOperand())) {
        return binaryExpression.rightOperand();
      }
    }
    if (condition.is(NOT) && isConstant(((UnaryExpression) condition).expression())) {
      return ((UnaryExpression) condition).expression();
    }
    return null;
  }

  private static boolean isConstantCollectionLiteral(Expression condition) {
    switch (condition.getKind()) {
      case LIST_LITERAL:
        return isAlwaysEmptyOrNonEmptyCollection(((ListLiteral) condition).elements().expressions());
      case DICTIONARY_LITERAL:
        return isAlwaysEmptyOrNonEmptyCollection(((DictionaryLiteral) condition).elements());
      case SET_LITERAL:
        return isAlwaysEmptyOrNonEmptyCollection(((SetLiteral) condition).elements());
      case TUPLE:
        return isAlwaysEmptyOrNonEmptyCollection(((Tuple) condition).elements());
      default:
        return false;
    }
  }

  private static boolean isAlwaysEmptyOrNonEmptyCollection(List<? extends Tree> elements) {
    if (elements.isEmpty()) {
      return true;
    }
    return elements.stream().anyMatch(element -> !element.is(UNPACKING_EXPR));
  }

  /**
   * Checks if boolean expressions are used as an alternative to 'Conditional Expression'.
   * e.g. 'x = f() or 3 or g()'
   * Note that one level of nesting is checked: deeply nested boolean expressions are ignored.
   */
  @Override
  public void visitBinaryExpression(BinaryExpression binaryExpression) {
    if (!binaryExpression.is(AND, OR)) {
      return;
    }
    if (isConstant(binaryExpression.leftOperand())) {
      addIssue(binaryExpression.leftOperand(), MESSAGE);
      return;
    }
    if (binaryExpression.leftOperand().is(AND, OR)) {
      BinaryExpression leftOperand = (BinaryExpression) binaryExpression.leftOperand();
      checkExpression(leftOperand.leftOperand());
      if (!(leftOperand.is(AND) && binaryExpression.is(OR))) {
        // avoid 'f() and 3 or g()'
        // no issue is raised here because '3' is the expression value when the first f() returns true.
        checkExpression(leftOperand.rightOperand());
      }
      return;
    }

    if (binaryExpression.rightOperand().is(AND, OR)) {
      checkExpression(((BinaryExpression) binaryExpression.rightOperand()).leftOperand());
    }
  }

  private void checkExpression(Expression expression) {
    if (isConstant(expression)) {
      addIssue(expression, MESSAGE);
      return;
    }
    if (expression.is(NAME) || expression.is(QUALIFIED_EXPR)) {
      Symbol symbol = ((HasSymbol) expression).symbol();
      if (symbol != null && isClassOrFunction(symbol)) {
        raiseIssueOnClassOrFunction(expression, symbol);
        return;
      }
    }
    if (expression.is(NAME)) {
      Set<Expression> valuesAtLocation = reachingDefinitionsAnalysis.valuesAtLocation(((Name) expression));
      if (valuesAtLocation.size() == 1) {
        Expression lastAssignedValue = valuesAtLocation.iterator().next();
        if (isImmutableConstant(lastAssignedValue)) {
          addIssue(expression, MESSAGE).secondary(lastAssignedValue, "Last assignment.");
        }
      }
    }
  }

  private void raiseIssueOnClassOrFunction(Expression expression, Symbol symbol) {
    PreciseIssue issue = addIssue(expression, MESSAGE);
    LocationInFile locationInFile = locationForClassOrFunction(symbol);
    if (locationInFile != null) {
      String type = symbol.is(Symbol.Kind.CLASS) ? "Class" : "Function";
      issue.secondary(locationInFile, String.format("%s definition.", type));
    }
  }

  private static boolean isClassOrFunction(Symbol symbol) {
    if (symbol.is(Symbol.Kind.CLASS)) {
      return true;
    }
    if (symbol.is(Symbol.Kind.FUNCTION)) {
      // Avoid potential FPs with properties: only report on limited selection of "safe" decorators
      return ACCEPTED_DECORATORS.containsAll(((FunctionSymbol) symbol).decorators());
    }
    return false;
  }

  private static LocationInFile locationForClassOrFunction(Symbol symbol) {
    return symbol.is(Symbol.Kind.CLASS) ? ((ClassSymbol) symbol).definitionLocation() : ((FunctionSymbol) symbol).definitionLocation();
  }
}
