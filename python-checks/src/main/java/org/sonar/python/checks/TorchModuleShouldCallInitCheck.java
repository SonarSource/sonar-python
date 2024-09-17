package org.sonar.python.checks;

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6978")
public class TorchModuleShouldCallInitCheck extends PythonSubscriptionCheck {

  public static final String TORCH_NN_MODULE = "torch.nn.Module";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef funcDef = (FunctionDef) ctx.syntaxNode();
      ClassDef classDef = CheckUtils.getParentClassDef(funcDef);
      if (isConstructor(funcDef) && isInheritingFromTorchModule(classDef) && isMissingSuperCall(funcDef)) {
        PreciseIssue issue = ctx.addIssue(funcDef.name(), "Add a call to super().__init__()");
        issue.secondary(classDef.name(), "Inheritance happens here");
      }
    });
  }

  private static boolean isConstructor(FunctionDef funcDef) {
    FunctionSymbol symbol = TreeUtils.getFunctionSymbolFromDef(funcDef);
    return symbol != null && "__init__".equals(symbol.name()) && funcDef.isMethodDefinition();
  }

  private static boolean isInheritingFromTorchModule(@Nullable ClassDef classDef) {
    if (classDef == null || classDef.args() == null) return false;
    return classDef.args().arguments().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .anyMatch(arg -> TORCH_NN_MODULE.equals(getQualifiedName(arg.expression())));
  }

  //TODO: Copied from NonStandardCryptographicCheck. maybe worth refactoring into some Utils class
  private static String getQualifiedName(Expression node) {
    if (node instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      return symbol != null ? symbol.fullyQualifiedName() : "";
    }
    return "";
  }

  private static boolean isMissingSuperCall(FunctionDef funcDef) {
    return !TreeUtils.hasDescendant(funcDef, t -> t.is(Tree.Kind.CALL_EXPR) && isSuperConstructorCall(((CallExpression) t)));
  }

  private static boolean isSuperConstructorCall(CallExpression callExpr) {
    return callExpr.callee() instanceof QualifiedExpression qualifiedCallee && isSuperCall(qualifiedCallee.qualifier()) && "__init__".equals(qualifiedCallee.name().name());
  }

  private static boolean isSuperCall(Expression qualifier) {
    if (qualifier instanceof CallExpression callExpression) {
      Symbol superSymbol = callExpression.calleeSymbol();
      return superSymbol != null && "super".equals(superSymbol.name());
    }
    return false;
  }
}
