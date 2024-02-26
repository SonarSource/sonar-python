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

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix.Builder;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S1244")
public class FloatingPointEqualityCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not perform equality checks with floating point values.";
  private static final String QUICK_FIX_MESSAGE = "Replace with \"%s%s.isclose()\".";

  private static final String QUICK_FIX_MATH = "%s%s.isclose(%s, %s, rel_tol=1e-09, abs_tol=1e-09)";

  private static final String QUICK_FIX_IMPORTED_MODULE = "%s%s.isclose(%s, %s, rtol=1e-09, atol=1e-09)";

  private static final Tree.Kind[] BINARY_OPERATION_KINDS = { Tree.Kind.PLUS, Tree.Kind.MINUS, Tree.Kind.MULTIPLICATION,
      Tree.Kind.DIVISION };

  private static final String MATH_MODULE = "math";

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;
  private static final List<String> SUPPORTED_IS_CLOSE_MODULES = Arrays.asList("numpy", "torch", MATH_MODULE);

  private String importedModuleForIsClose;
  private Name importedAlias;
  private boolean isMathImported = false;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeAnalysis);

    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME,
        ctx -> ((ImportName) ctx.syntaxNode()).modules().forEach(this::addImportedName));

    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, this::checkFloatingPointEquality);
  }

  private void initializeAnalysis(SubscriptionContext ctx) {
    reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile());
    importedModuleForIsClose = null;
    importedAlias = null;
  }

  private void checkFloatingPointEquality(SubscriptionContext ctx) {
    BinaryExpression binaryExpression = (BinaryExpression) ctx.syntaxNode();
    String operator = binaryExpression.operator().value();
    if (("==".equals(operator) || "!=".equals(operator)) && isAnyOperandFloatingPoint(binaryExpression)) {
      PreciseIssue issue = ctx.addIssue(binaryExpression, MESSAGE);
      issue.addQuickFix(createQuickFix(binaryExpression, operator));
    }
  }

  private boolean isAnyOperandFloatingPoint(BinaryExpression binaryExpression) {
    Expression leftOperand = binaryExpression.leftOperand();
    Expression rightOperand = binaryExpression.rightOperand();

    return isFloat(leftOperand) || isFloat(rightOperand) ||
        isAssignedFloat(leftOperand) || isAssignedFloat(rightOperand) ||
        isBinaryOperationWithFloat(leftOperand) || isBinaryOperationWithFloat(rightOperand);
  }

  private static boolean isFloat(Expression expression) {
    return expression.is(Tree.Kind.NUMERIC_LITERAL) && expression.type().equals(InferredTypes.FLOAT);
  }

  private boolean isAssignedFloat(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Set<Expression> values = reachingDefinitionsAnalysis.valuesAtLocation((Name) expression);
      if (!values.isEmpty()) {
        return values.stream().allMatch(value -> isFloat(value));
      }
    }
    return false;
  }

  private boolean isBinaryOperationWithFloat(Expression expression) {
    if (expression.is(BINARY_OPERATION_KINDS)) {
      return isAnyOperandFloatingPoint((BinaryExpression) expression);
    }
    return false;
  }

  private PythonQuickFix createQuickFix(BinaryExpression binaryExpression, String operator) {
    String notToken = "!=".equals(operator) ? "not " : "";
    String isCloseModuleName = getModuleNameOrAliasForIsClose();
    String message = String.format(QUICK_FIX_MESSAGE, notToken, isCloseModuleName);
    Builder quickFix = PythonQuickFix.newQuickFix(message);

    String quickFixText = MATH_MODULE.equals(isCloseModuleName) ? QUICK_FIX_MATH : QUICK_FIX_IMPORTED_MODULE;
    String quickFixTextWithModuleName = String.format(quickFixText, notToken, isCloseModuleName,
        TreeUtils.treeToString(binaryExpression.leftOperand(), false),
        TreeUtils.treeToString(binaryExpression.rightOperand(), false));

    quickFix.addTextEdit(TextEditUtils.replace(binaryExpression, quickFixTextWithModuleName));

    if (MATH_MODULE.equals(isCloseModuleName) && !isMathImported) {
      quickFix.addTextEdit(TextEditUtils.insertAtPosition(0, 0, "import math\n"));
    }

    return quickFix.build();
  }

  private String getModuleNameOrAliasForIsClose() {
    if (importedAlias != null) {
      return importedAlias.name();
    }
    if (importedModuleForIsClose != null) {
      return importedModuleForIsClose;
    }
    return MATH_MODULE;
  }

  private void addImportedName(AliasedName aliasedName) {
    List<Name> importNames = aliasedName.dottedName().names();
    if (importedModuleForIsClose == null || MATH_MODULE.equals(importedModuleForIsClose)) {
      importNames.stream()
          .filter(name -> SUPPORTED_IS_CLOSE_MODULES.contains(name.name()))
          .findFirst()
          .map(Name::name)
          .ifPresent(name -> {
            if(MATH_MODULE.equals(name)){
              isMathImported = true;
            }
            importedModuleForIsClose = name;
            importedAlias = aliasedName.alias();
          });
    }
  }
}
