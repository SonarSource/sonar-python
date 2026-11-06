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

import java.util.IdentityHashMap;
import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_GROUP_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FINALLY_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.FUNCDEF;
import static org.sonar.plugins.python.api.tree.Tree.Kind.RAISE_STMT;

@Rule(key = "S5747")
public class RaiseOutsideExceptCheck extends PythonSubscriptionCheck {
  private final Map<FunctionDef, Boolean> methodCallsInsideExceptBlock = new IdentityHashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(RAISE_STMT, ctx -> {
      RaiseStatement raiseStatement = (RaiseStatement) ctx.syntaxNode();
      if (!raiseStatement.expressions().isEmpty()) {
        return;
      }
      if (!isInsideExceptOrFinally(raiseStatement)) {
        ctx.addIssue(raiseStatement, "Remove this \"raise\" statement or move it inside an \"except\" block.");
      }

    });
  }

  @Override
  public void leaveFile() {
    methodCallsInsideExceptBlock.clear();
  }

  private boolean isInsideExceptOrFinally(Tree tree) {
    Tree parent = tree.parent();
    while (parent != null) {
      if ((parent.is(FUNCDEF))) {
        // __exit__ special method is handled by S5706
        return "__exit__".equals(((FunctionDef) parent).name().name()) || isFunctionCalledInsideExceptBlock(((FunctionDef) parent));
      }
      if (parent.is(EXCEPT_CLAUSE, EXCEPT_GROUP_CLAUSE, FINALLY_CLAUSE)) {
        return true;
      }
      parent = parent.parent();
    }
    return false;
  }

  private boolean isFunctionCalledInsideExceptBlock(FunctionDef functionDef) {
    SymbolV2 symbol = functionDef.name().symbolV2();
    boolean hasUsageInsideExceptBlock = symbol != null && symbol.usages().stream()
      .filter(usage -> usage.tree() != functionDef.name())
      .anyMatch(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), EXCEPT_CLAUSE, EXCEPT_GROUP_CLAUSE, FINALLY_CLAUSE) != null);
    return hasUsageInsideExceptBlock || methodCallsInsideExceptBlock.computeIfAbsent(functionDef, RaiseOutsideExceptCheck::isMethodCalledInsideExceptBlock);
  }

  private static boolean isMethodCalledInsideExceptBlock(FunctionDef functionDef) {
    ClassDef classDef = CheckUtils.getParentClassDef(functionDef);
    if (classDef == null || !functionDef.isMethodDefinition()) {
      return false;
    }
    Tree compilationUnit = classDef;
    while (compilationUnit.parent() != null) {
      compilationUnit = compilationUnit.parent();
    }
    var visitor = new MethodCallInsideExceptVisitor(functionDef.name().name(), classDef);
    return visitor.isCalledInsideExceptBlock(compilationUnit);
  }

  private static class MethodCallInsideExceptVisitor extends BaseTreeVisitor {
    private final String methodName;
    private final ClassDef classDef;
    private final SymbolV2 classSymbol;
    private boolean calledInsideExceptBlock;
    private SymbolV2 receiverSymbol;

    MethodCallInsideExceptVisitor(String methodName, ClassDef classDef) {
      this.methodName = methodName;
      this.classDef = classDef;
      this.classSymbol = classDef.name().symbolV2();
    }

    boolean isCalledInsideExceptBlock(Tree tree) {
      scan(tree);
      return calledInsideExceptBlock;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
        QualifiedExpression callee = (QualifiedExpression) callExpression.callee();
        SymbolV2 qualifierSymbol = callee.qualifier() instanceof Name name ? name.symbolV2() : null;
        if (qualifierSymbol != null && methodName.equals(callee.name().name())
          && (qualifierSymbol == receiverSymbol || qualifierSymbol == classSymbol)
          && TreeUtils.firstAncestorOfKind(callExpression, EXCEPT_CLAUSE, EXCEPT_GROUP_CLAUSE, FINALLY_CLAUSE) != null) {
          calledInsideExceptBlock = true;
        }
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      SymbolV2 previousReceiverSymbol = receiverSymbol;
      if (functionDef.isMethodDefinition()) {
        receiverSymbol = isMethodOfClass(functionDef) ? receiverSymbol(functionDef) : null;
      }
      super.visitFunctionDef(functionDef);
      receiverSymbol = previousReceiverSymbol;
    }

    @Override
    public void visitClassDef(ClassDef nestedClassDef) {
      SymbolV2 previousReceiverSymbol = receiverSymbol;
      receiverSymbol = null;
      super.visitClassDef(nestedClassDef);
      receiverSymbol = previousReceiverSymbol;
    }

    private boolean isMethodOfClass(FunctionDef functionDef) {
      return functionDef.isMethodDefinition()
        && CheckUtils.getParentClassDef(functionDef) == classDef
        && functionDef.decorators().stream().noneMatch(decorator -> "staticmethod".equals(TreeUtils.decoratorNameFromExpression(decorator.expression())));
    }

    private static SymbolV2 receiverSymbol(FunctionDef functionDef) {
      if (functionDef.parameters().all().isEmpty() || !functionDef.parameters().all().get(0).is(Tree.Kind.PARAMETER)) {
        return null;
      }
      Name receiver = ((Parameter) functionDef.parameters().all().get(0)).name();
      return receiver != null ? receiver.symbolV2() : null;
    }
  }
}
