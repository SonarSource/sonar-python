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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8494")
public class SlotsAssignmentCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add \"%s\" to the class's \"__slots__\".";

  /**
   * Built-in C types whose instances don't have __dict__ by default.
   * When subclassing these, __slots__ restrictions still apply even though
   * we can't find a ClassDef to inspect.
   */
  private static final Set<String> BUILTIN_TYPES_WITHOUT_DICT = Set.of(
    BuiltinTypes.INT, BuiltinTypes.FLOAT, BuiltinTypes.COMPLEX,
    BuiltinTypes.STR, BuiltinTypes.BYTES,
    BuiltinTypes.BOOL, BuiltinTypes.LIST, BuiltinTypes.TUPLE,
    BuiltinTypes.SET, BuiltinTypes.DICT, BuiltinTypes.NONE_TYPE,
    "type", "frozenset", "memoryview", "bytearray", "range");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, SlotsAssignmentCheck::checkClass);
  }

  private static void checkClass(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();
    PythonType classType = classDef.name().typeV2();
    if (!(classType instanceof ClassType classTypeV2)) {
      return;
    }

    // Find __slots__ assignment in this class body
    Set<String> ownSlots = extractOwnSlots(classDef);
    if (ownSlots == null) {
      // No __slots__ found or dynamic/unsupported form
      return;
    }

    if (ownSlots.contains("__dict__")) {
      // Explicit opt-out: __dict__ in slots means no restriction
      return;
    }

    if (classTypeV2.hasUnresolvedHierarchy()) {
      // Can't safely analyze hierarchy
      return;
    }

    // Build allowed slots: own + ancestors
    Set<String> allowedSlots = new HashSet<>(ownSlots);
    Set<ClassType> visited = new HashSet<>();
    if (!collectAncestorSlots(classTypeV2, classDef, allowedSlots, visited)) {
      // Some resolvable parent has no __slots__, so __dict__ is available
      return;
    }

    // Check each instance method
    String className = classDef.name().name();
    for (FunctionDef functionDef : TreeUtils.topLevelFunctionDefs(classDef)) {
      PythonType funcType = functionDef.name().typeV2();
      if (!(funcType instanceof FunctionType ft) || !ft.isInstanceMethod()) {
        continue;
      }
      List<Parameter> params = TreeUtils.positionalParameters(functionDef);
      if (!params.isEmpty()) {
        String selfName = params.get(0).name().name();
        SelfAttributeAssignmentVisitor visitor = new SelfAttributeAssignmentVisitor(selfName, allowedSlots, className, ctx);
        functionDef.body().accept(visitor);
      }
    }
  }

  /**
   * Extracts the slot names from the __slots__ assignment in the class body.
   * Uses the last __slots__ assignment, since Python's class body executes top-to-bottom
   * and the metaclass reads __slots__ from the final namespace value.
   * Returns null if no __slots__ is found or if the form is not a supported literal.
   * Returns an empty set if __slots__ = [] or __slots__ = ().
   */
  @Nullable
  private static Set<String> extractOwnSlots(ClassDef classDef) {
    Set<String> result = null;
    for (Statement stmt : classDef.body().statements()) {
      if (stmt instanceof AssignmentStatement assignment) {
        List<ExpressionList> lhsExpressions = assignment.lhsExpressions();
        if (lhsExpressions.size() == 1) {
          List<Expression> lhs = lhsExpressions.get(0).expressions();
          if (lhs.size() == 1 && lhs.get(0).is(Tree.Kind.NAME) && "__slots__".equals(((Name) lhs.get(0)).name())) {
            // Found __slots__ assignment; keep iterating to use the last one
            result = extractSlotNames(assignment.assignedValue());
          }
        }
      }
    }
    return result;
  }

  /**
   * Extracts slot names from the RHS of a __slots__ assignment.
   * Returns null if the form is not a supported literal.
   */
  @Nullable
  private static Set<String> extractSlotNames(Expression value) {
    List<Expression> elements;
    if (value instanceof ListLiteral listLiteral) {
      elements = listLiteral.elements().expressions();
    } else if (value instanceof Tuple tuple) {
      elements = tuple.elements();
    } else if (value instanceof SetLiteral setLiteral) {
      elements = setLiteral.elements();
    } else if (value instanceof DictionaryLiteral dictionaryLiteral) {
      elements = extractDictKeys(dictionaryLiteral);
    } else if (value instanceof StringLiteral stringLiteral) {
      // Single string: __slots__ = 'attr_name'
      return Set.of(stringLiteral.trimmedQuotesValue());
    } else {
      // Dynamic or unsupported form
      return null;
    }
    return elements != null ? extractStringLiterals(elements) : null;
  }

  /**
   * Extracts key expressions from a dictionary literal.
   * Returns null if any element is not a KeyValuePair with a string key (e.g. unpacking).
   */
  @Nullable
  private static List<Expression> extractDictKeys(DictionaryLiteral dictionaryLiteral) {
    List<Expression> keys = new ArrayList<>();
    for (Tree element : dictionaryLiteral.elements()) {
      if (!(element instanceof KeyValuePair kvp)) {
        return null;
      }
      keys.add(kvp.key());
    }
    return keys;
  }

  /**
   * Collects trimmed string values from a list of expressions.
   * If an element is a Name, attempts to resolve it to a StringLiteral via single-assignment analysis.
   * Returns null if any element cannot be resolved to a string literal.
   */
  @Nullable
  private static Set<String> extractStringLiterals(List<Expression> elements) {
    Set<String> slots = new HashSet<>();
    for (Expression element : elements) {
      if (element instanceof StringLiteral stringLiteral) {
        slots.add(stringLiteral.trimmedQuotesValue());
      } else if (element instanceof Name name) {
        Optional<Expression> resolved = Expressions.singleAssignedNonNameValue(name);
        if (resolved.isPresent() && resolved.get() instanceof StringLiteral stringLiteral) {
          slots.add(stringLiteral.trimmedQuotesValue());
        } else {
          return null;
        }
      } else {
        return null;
      }
    }
    return slots;
  }

  /**
   * Collects slot names from all ancestor classes into allowedSlots.
   * Uses ClassType (v2) for type hierarchy filtering (FQN, object skipping) and
   * resolves parent ClassDef AST nodes via SymbolV2's CLASS_DECLARATION usages.
   * Returns false if any resolvable parent has no __slots__ (meaning __dict__ is available).
   */
  private static boolean collectAncestorSlots(ClassType classType, ClassDef classDef, Set<String> allowedSlots,
    Set<ClassType> visited) {
    for (TypeWrapper parentWrapper : classType.superClasses()) {
      PythonType parentType = parentWrapper.type();
      if (!(parentType instanceof ClassType parentClassType)
        || "object".equals(parentClassType.fullyQualifiedName())
        || !visited.add(parentClassType)) {
        continue;
      }
      if (!processParentType(parentClassType, classDef, allowedSlots, visited)) {
        return false;
      }
    }
    return true;
  }

  private static boolean processParentType(ClassType parentClassType, ClassDef classDef, Set<String> allowedSlots,
    Set<ClassType> visited) {
    Optional<ClassDef> parentClassDef = findParentClassDef(classDef, parentClassType);
    if (parentClassDef.isEmpty()) {
      // Can't inspect the parent's __slots__. Built-in C types don't provide __dict__
      // on their instances, so slots restrictions still apply when subclassing them.
      // For all other types, we can't determine if __dict__ is available, so bail out.
      String fqn = parentClassType.fullyQualifiedName();
      return fqn != null && BUILTIN_TYPES_WITHOUT_DICT.contains(fqn);
    }
    return processParentClass(parentClassDef.get(), allowedSlots, visited);
  }

  private static Optional<ClassDef> findParentClassDef(ClassDef childClassDef, ClassType parentClassType) {
    if (childClassDef.args() == null) {
      return Optional.empty();
    }
    for (var arg : childClassDef.args().arguments()) {
      if (arg instanceof RegularArgument regArg) {
        Expression expr = regArg.expression();
        if (expr instanceof Name name && name.typeV2() == parentClassType) {
          return findClassDefFromSymbolV2(name.symbolV2());
        }
      }
    }
    return Optional.empty();
  }

  /**
   * Finds the ClassDef AST node from a SymbolV2 by looking for its CLASS_DECLARATION usage.
   */
  private static Optional<ClassDef> findClassDefFromSymbolV2(@Nullable SymbolV2 symbolV2) {
    if (symbolV2 == null) {
      return Optional.empty();
    }
    return symbolV2.usages().stream()
      .filter(usage -> usage.kind() == UsageV2.Kind.CLASS_DECLARATION)
      .map(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.CLASSDEF))
      .filter(Objects::nonNull)
      .map(ClassDef.class::cast)
      .findFirst();
  }

  private static boolean processParentClass(ClassDef parentClassDef, Set<String> allowedSlots, Set<ClassType> visited) {
    Set<String> parentSlots = extractOwnSlots(parentClassDef);
    if (parentSlots == null || parentSlots.contains("__dict__")) {
      return false;
    }

    allowedSlots.addAll(parentSlots);

    PythonType parentType = parentClassDef.name().typeV2();
    if (!(parentType instanceof ClassType parentClassType)) {
      return true;
    }
    return !parentClassType.hasUnresolvedHierarchy() && collectAncestorSlots(parentClassType, parentClassDef, allowedSlots, visited);
  }

  private static class SelfAttributeAssignmentVisitor extends BaseTreeVisitor {
    private final String selfName;
    private final Set<String> allowedSlots;
    private final String className;
    private final SubscriptionContext ctx;

    SelfAttributeAssignmentVisitor(String selfName, Set<String> allowedSlots, String className, SubscriptionContext ctx) {
      this.selfName = selfName;
      this.allowedSlots = allowedSlots;
      this.className = className;
      this.ctx = ctx;
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignment) {
      for (ExpressionList exprList : assignment.lhsExpressions()) {
        for (Expression expr : exprList.expressions()) {
          checkQualifiedExpression(expr);
        }
      }
      // Do not call super to avoid visiting RHS as assignment targets
    }

    @Override
    public void visitCompoundAssignment(CompoundAssignmentStatement compoundAssignment) {
      checkQualifiedExpression(compoundAssignment.lhsExpression());
      // Do not call super
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // Stop recursion into nested functions to avoid false positives
      // where an inner function's first param shadows self
    }

    @Override
    public void visitClassDef(ClassDef classDef) {
      // Stop recursion into nested classes to avoid false positives
    }

    private void checkQualifiedExpression(Expression expr) {
      if (!expr.is(Tree.Kind.QUALIFIED_EXPR)) {
        return;
      }
      QualifiedExpression qualifiedExpr = (QualifiedExpression) expr;
      Expression qualifier = qualifiedExpr.qualifier();
      if (!qualifier.is(Tree.Kind.NAME)) {
        return;
      }
      if (!selfName.equals(((Name) qualifier).name())) {
        return;
      }
      String attrName = qualifiedExpr.name().name();
      if (!allowedSlots.contains(attrName) && !allowedSlots.contains(mangledName(attrName))) {
        ctx.addIssue(qualifiedExpr.name(), String.format(MESSAGE, attrName));
      }
    }

    /**
     * Returns the name-mangled form of a private attribute name within this class.
     * Python mangles names starting with two underscores (but not ending with two underscores)
     * by stripping leading underscores from the class name and prepending "_ClassName".
     * For example, "__den" in class "_Rat" becomes "_Rat__den" (not "__Rat__den").
     * See: https://docs.python.org/3/reference/expressions.html#atom-identifiers
     */
    private String mangledName(String attrName) {
      if (attrName.startsWith("__") && !attrName.endsWith("__")) {
        return "_" + className.replaceAll("^_+", "") + attrName;
      }
      return attrName;
    }
  }
}
