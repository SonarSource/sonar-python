package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S5886")
public class FunctionReturnTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Return a \"%s\" instead of a \"%s\" or update function \"%s\" type hint.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      Symbol symbol = functionDef.name().symbol();
      if (!symbol.is(Symbol.Kind.FUNCTION)) {
        return;
      }
      FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) symbol;
      InferredType declaredReturnType = functionSymbol.declaredReturnType();
      if (declaredReturnType == InferredTypes.anyType()) {
        return;
      }
      ReturnTypeVisitor returnTypeVisitor = new ReturnTypeVisitor(declaredReturnType);
      functionDef.accept(returnTypeVisitor);
      if (returnTypeVisitor.containsYield && declaredReturnType.canOnlyBe("typing.Generator")) {
        // FIXME: Avoid FP for generators (should use mustBeOrExtend)
        return;
      }
      returnTypeVisitor.invalidReturnsOrYield.forEach(i -> {
        String functionName = functionDef.name().name();
        String returnTypeName = InferredTypes.typeName(declaredReturnType);
        if (i.expressions().size() > 1) {
          ctx.addIssue(i, String.format(MESSAGE, returnTypeName, "tuple", functionName));
        } else if (i.expressions().size() == 1 && InferredTypes.typeName(i.expressions().get(0).type()) != null) {
          ctx.addIssue(i.expressions().get(0), String.format(MESSAGE, returnTypeName,
            InferredTypes.typeName(i.expressions().get(0).type()), functionName));
        } else {
          ctx.addIssue(i, String.format("Return a \"%s\" or update function \"%s\" type hint.", returnTypeName, functionName));
        }
      });
    });
  }

  private static class ReturnTypeVisitor extends BaseTreeVisitor {

    InferredType returnType;
    boolean containsYield = false;
    List<ReturnStatement> invalidReturnsOrYield = new ArrayList<>();

    ReturnTypeVisitor(InferredType returnType) {
      this.returnType = returnType;
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      List<Expression> expressions = returnStatement.expressions();
      if (expressions.isEmpty()) {
        // check if can return None
        //FIXME: should use "mustBeOrExtend"
        if (!returnType.canBeOrExtend("NoneType")) {
          invalidReturnsOrYield.add(returnStatement);
        }
      } else if (expressions.size() > 1) {
        // check if type hint is tuple?
        //FIXME: should use "mustBeOrExtend"
        if (!returnType.canOnlyBe("tuple")) {
          invalidReturnsOrYield.add(returnStatement);
        }
      } else {
        Expression expression = expressions.get(0);
        InferredType inferredType = expression.type();
        if (!inferredType.isCompatibleWith(returnType)) {
          invalidReturnsOrYield.add(returnStatement);
        }
      }
      super.visitReturnStatement(returnStatement);
    }

    @Override
    public void visitYieldStatement(YieldStatement pyYieldStatementTree) {
      containsYield = true;
      super.visitYieldStatement(pyYieldStatementTree);
    }
  }
}
