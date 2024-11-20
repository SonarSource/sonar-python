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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;

import static org.sonar.plugins.python.api.tree.Tree.Kind.AND;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NOT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.OR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.QUALIFIED_EXPR;
import static org.sonar.python.checks.utils.CheckUtils.isConstant;
import static org.sonar.python.checks.utils.CheckUtils.isImmutableConstant;

@Rule(key = "S5797")
public class ConstantConditionCheck extends PythonVisitorCheck {

  private static final String MESSAGE = "Replace this expression; used as a condition it will always be constant.";
  private static final List<String> ACCEPTED_DECORATORS = List.of("overload", "staticmethod", "classmethod");
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
