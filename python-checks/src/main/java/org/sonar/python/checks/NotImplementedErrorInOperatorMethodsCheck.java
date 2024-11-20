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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key="S5712")
public class NotImplementedErrorInOperatorMethodsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Return \"NotImplemented\" instead of raising \"NotImplementedError\"";
  private static final String NOT_IMPLEMENTED_ERROR = "NotImplementedError";

  private static final List<String> OPERATOR_METHODS = Arrays.asList(
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rmatmul__",
    "__rtruediv__",
    "__rfloordiv__",
    "__rmod__",
    "__rdivmod__",
    "__rpow__",
    "__rlshift__",
    "__rrshift__",
    "__rand__",
    "__rxor__",
    "__ror__",
    "__iadd__",
    "__isub__",
    "__imul__",
    "__imatmul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
    "__length_hint__"
  );
  public static final String QUICK_FIX_MESSAGE = "Replace the raised exception with return NotImplemented";

  private static class RaiseNotImplementedErrorVisitor extends BaseTreeVisitor {
    private List<RaiseStatement> nonCompliantRaises = new ArrayList<>();

    @Override
    public void visitRaiseStatement(RaiseStatement pyRaiseStatementTree) {
      if (pyRaiseStatementTree.expressions().isEmpty()) {
        // Do not bother with bare raises.
        return;
      }

      Expression raisedException = pyRaiseStatementTree.expressions().get(0);
      if (raisedException.type().canOnlyBe(NOT_IMPLEMENTED_ERROR)) {
        nonCompliantRaises.add(pyRaiseStatementTree);
      } else if (raisedException instanceof HasSymbol hasSymbol) {
        Symbol symbol = hasSymbol.symbol();
        if (symbol != null && NOT_IMPLEMENTED_ERROR.equals(symbol.fullyQualifiedName())) {
          nonCompliantRaises.add(pyRaiseStatementTree);
        }
      }
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (!functionDef.isMethodDefinition() || !OPERATOR_METHODS.contains(functionDef.name().name())) {
        return;
      }

      RaiseNotImplementedErrorVisitor visitor = new RaiseNotImplementedErrorVisitor();
      functionDef.accept(visitor);

      for (RaiseStatement notImplementedErrorRaise : visitor.nonCompliantRaises) {
        var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(notImplementedErrorRaise, "return NotImplemented"))
          .build();
        var issue = ctx.addIssue(notImplementedErrorRaise, MESSAGE);
        issue.addQuickFix(quickFix);
      }
    });
  }
}
