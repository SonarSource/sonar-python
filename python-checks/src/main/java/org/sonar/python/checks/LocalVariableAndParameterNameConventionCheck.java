/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S117")
public class LocalVariableAndParameterNameConventionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename this %s \"%s\" to match the regular expression %s.";
  private static final String PARAMETER = "parameter";
  private static final String LOCAL_VAR = "local variable";
  private static final EnumSet<UsageV2.Kind> USAGES = EnumSet.of(UsageV2.Kind.PARAMETER, UsageV2.Kind.LOOP_DECLARATION, UsageV2.Kind.ASSIGNMENT_LHS);

  private static final String CONSTANT_PATTERN = "^[_A-Z][A-Z0-9_]*$";

  private static final String DEFAULT = "^[_a-z][a-z0-9_]*$";
  @RuleProperty(
    key = "format",
    description = "Regular expression used to check the names against.",
    defaultValue = DEFAULT)
  public String format = DEFAULT;
  private Pattern constantPattern;
  private Pattern pattern;
  private static final Set<String> ML_VARIABLE_NAMES = Set.of("X_train", "X_test", "Y_train", "Y_test", "X", "Y");

  private TypeCheckBuilder isDjangoModelTypeCheck;

  @Override
  public void initialize(Context context) {
    pattern = Pattern.compile(format);
    constantPattern = Pattern.compile(CONSTANT_PATTERN);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initializeTypeChecker);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkFunctionDef);
  }

  private void initializeTypeChecker(SubscriptionContext ctx) {
    isDjangoModelTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeOrInstanceWithName("type");
  }

  private void checkFunctionDef(SubscriptionContext ctx) {
    FunctionDef funcDef = (FunctionDef) ctx.syntaxNode();
    TreeUtils.getLocalVariableSymbols(funcDef).stream()
      .sorted(Comparator.comparing(SymbolV2::name))
      .forEach(s -> checkName(s, ctx));
  }

  private void checkName(SymbolV2 symbol, SubscriptionContext ctx) {
    String name = symbol.name();
    if (ML_VARIABLE_NAMES.contains(name)) {
      return;
    }
    if (!pattern.matcher(name).matches()) {
      if (isType(symbol)) {
        // Type variables generally adhere to class naming conventions rather than regular variable naming conventions
        return;
      }
      symbol.usages().stream()
        .filter(usage -> USAGES.contains(usage.kind()))
        .sorted(Comparator.comparingInt(u -> u.tree().firstToken().line()))
        .limit(1)
        .forEach(usage -> raiseIssueForNameAndUsage(ctx, name, usage));
    }
  }

  private boolean isType(SymbolV2 symbolV2) {
    // TypeV1 and TypeV2 can detect different cases and work complementary to find more issues
    Symbol symbolV1 = SymbolUtils.symbolV2ToSymbolV1(symbolV2).orElse(null);
    return symbolV1 != null && (isExtendingType(symbolV1) || isAssignedFromTyping(symbolV2) || isPythonTypeAClassType(symbolV2));
  }

  private static boolean isExtendingType(Symbol symbol) {
    boolean isInferredTypeExtendingType = symbol.usages().stream()
      .map(Usage::tree)
      .filter(Expression.class::isInstance)
      .map(Expression.class::cast)
      .anyMatch(e -> e.type().mustBeOrExtend("type"));

    boolean isAnnotatedTypeStartingWithTyping = symbol.annotatedTypeName() != null && symbol.annotatedTypeName().startsWith("typing.");
    return isInferredTypeExtendingType || isAnnotatedTypeStartingWithTyping;
  }

  private static boolean isAssignedFromTyping(SymbolV2 symbol) {
    List<Expression> assignedValues = symbol.usages().stream()
      .filter(u -> u.kind() == UsageV2.Kind.ASSIGNMENT_LHS)
      .flatMap(usage -> getAssignedValue(usage.tree()))
      .toList();

    for (Expression assignedValue : assignedValues) {
      Symbol assignedSymbol = getTypingSymbol(assignedValue);
      if (assignedSymbol != null && isExtendingType(assignedSymbol)) {
        return true;
      }
    }
    return false;
  }

  private boolean isPythonTypeAClassType(SymbolV2 symbol) {
    PythonType type = SymbolUtils.getPythonType(symbol);
    return isDjangoModelTypeCheck.check(type).isTrue();
  }

  private static Stream<Expression> getAssignedValue(Tree assignmentName) {
    var assignmentStmt = TreeUtils.firstAncestorOfClass(assignmentName, AssignmentStatement.class);
    if (assignmentStmt != null) {
      return Stream.of(assignmentStmt.assignedValue());
    } else {
      return Stream.empty();
    }
  }

  private static @Nullable Symbol getTypingSymbol(Expression expr) {
    if (expr instanceof SubscriptionExpression subscriptionExpression
      && subscriptionExpression.object() instanceof Name name) {
      return name.symbol();
    }
    return null;
  }

  private void raiseIssueForNameAndUsage(SubscriptionContext ctx, String name, UsageV2 usage) {
    String type = PARAMETER;
    UsageV2.Kind kind = usage.kind();
    if (kind == UsageV2.Kind.ASSIGNMENT_LHS) {
      type = LOCAL_VAR;
      if (constantPattern.matcher(name).matches()) {
        return;
      }
    } else if (kind == UsageV2.Kind.LOOP_DECLARATION) {
      type = LOCAL_VAR;
      if (name.length() <= 1) {
        return;
      }
    } else if (kind == UsageV2.Kind.PARAMETER && isParameterNameFromOverriddenMethod(usage, name)) {
      return;
    }
    ctx.addIssue(usage.tree(), String.format(MESSAGE, type, name, format));
  }

  private static boolean isParameterNameFromOverriddenMethod(UsageV2 usage, String parameterName) {
    Tree functionLikeAncestor = TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA);
    if (functionLikeAncestor == null || functionLikeAncestor.is(Tree.Kind.LAMBDA)) {
      return false;
    }
    FunctionDef functionDef = (FunctionDef) functionLikeAncestor;

    int parameterIndex = getParameterIndex(functionDef, parameterName);
    if (parameterIndex < 0) {
      return false;
    }

    FunctionSymbol functionSymbol = TreeUtils.getFunctionSymbolFromDef(functionDef);
    if (functionSymbol == null) {
      return false;
    }

    List<FunctionSymbol> overriddenMethods = SymbolUtils.getOverriddenMethods(functionSymbol);
    for (FunctionSymbol overriddenMethod : overriddenMethods) {
      List<FunctionSymbol.Parameter> overriddenParams = overriddenMethod.parameters();
      // parameterIndex is based on positional parameters only, while overriddenParams contains all parameters.
      // This comparison works because positional parameters are always a prefix of the full parameter list.
      if (parameterIndex < overriddenParams.size()) {
        String overriddenParamName = overriddenParams.get(parameterIndex).name();
        if (parameterName.equals(overriddenParamName)) {
          return true;
        }
      }
    }
    return false;
  }

  private static int getParameterIndex(FunctionDef functionDef, String parameterName) {
    List<Parameter> parameters = TreeUtils.positionalParameters(functionDef);
    for (int i = 0; i < parameters.size(); i++) {
      Name paramName = parameters.get(i).name();
      if (paramName != null && parameterName.equals(paramName.name())) {
        return i;
      }
    }
    return -1;
  }
}
