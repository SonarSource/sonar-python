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
package org.sonar.python.types;

import com.google.common.annotations.VisibleForTesting;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.Scope;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.pytype.PyTypeInfo;
import org.sonar.python.types.pytype.PyTypeTable;
import org.sonar.python.types.pytype.json.PyTypeTableReader;

public class TypeContext {
  private static final Logger LOG = LoggerFactory.getLogger(TypeContext.class);
  public static final String CALLABLE_TYPE_CALLABLE = "CallableType(base_type=ClassType(typing.Callable)";
  public static final String GENERIC_TYPE_CALLABLE = "GenericType(base_type=ClassType(typing.Callable)";
  public static final ClassSymbolImpl CALLABLE_CLASS_SYMBOL = new ClassSymbolImpl("Callable", "typing.Callable");
  public static ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

  public void setProjectLevelSymbolTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
    TypeContext.projectLevelSymbolTable = projectLevelSymbolTable;
  }
  private PyTypeTable pyTypeTable;
  private Map<Tree, Scope> scopesByRootTree = null;

  public TypeContext() {
    pyTypeTable = new PyTypeTable(new HashMap<>());
  }

  public TypeContext(PyTypeTable pyTypeTable) {
    this.pyTypeTable = pyTypeTable;
  }

  public static TypeContext fromJson(String json) {
    return new TypeContext(PyTypeTableReader.fromJsonString(json));
  }

  public void setScopesByRootTree(Map<Tree, Scope> scopesByRootTree) {
    this.scopesByRootTree = scopesByRootTree;
  }

  @VisibleForTesting
  Optional<InferredType> getTypeFor(String fileName, int line, int column, String name, String kind, Tree tree) {
    return pyTypeTable.getVariableTypeFor(fileName, line, column, name, kind)
      .map(typeInfo -> getInferredType(typeInfo, fileName, tree));
  }

  private InferredType getInferredType(PyTypeInfo typeInfo, String fileName, Tree tree) {
    var typeString = typeInfo.shortType();
    var detailedType = typeInfo.type().raw();
    if ("None".equals(typeString)) {
      typeString = "NoneType";
    }
    if (tree != null) {
      Optional<InferredType> localType = getLocallyDefinedClassSymbolType(typeString, fileName, tree);
      if (localType.isPresent() && localType.filter(type -> InferredTypes.anyType().equals(type)).isEmpty()) {
        return localType.get();
      }
    }
    if (detailedType.startsWith(CALLABLE_TYPE_CALLABLE)) {
      return new RuntimeType(CALLABLE_CLASS_SYMBOL);
    }
    if (typeString.startsWith("Any")) {
      return InferredTypes.anyType();
    }
    // workaround until Typeshed is fixed
    try {
      PyTypeTypeGrammar.projectLevelSymbolTable = TypeContext.projectLevelSymbolTable;
      PyTypeTypeGrammar.fileName = fileName;
      return typeInfo.inferredType();
    } catch (Exception e) {
      // LOG.error("");
      // LOG.error(e.toString());
      // LOG.error(detailedType);
      // LOG.error("");
      return InferredTypes.anyType();
    }
    // return getInferredTypeWithoutParsing(typeString, detailedType, fileName);
  }

  private static InferredType getInferredTypeWithoutParsing(String typeString, String detailedType, String fileName) {
    if (detailedType.startsWith(CALLABLE_TYPE_CALLABLE)
      || detailedType.startsWith(GENERIC_TYPE_CALLABLE)) {
      return new RuntimeType(CALLABLE_CLASS_SYMBOL);
    } else if (detailedType.startsWith("builtins.") ||
      detailedType.startsWith("GenericType(base_type=ClassType(builtins.") ||
      detailedType.startsWith("TupleType(base_type=ClassType(builtins.")) {
        return InferredTypes.runtimeBuiltinType(getBaseType(typeString));
      } else if (typeString.indexOf('.') >= 0 || !detailedType.startsWith(typeString)) {
        ClassSymbolImpl callableClassSymbol = new ClassSymbolImpl(typeString.substring(typeString.lastIndexOf('.') + 1), detailedType);
        return new RuntimeType(callableClassSymbol);
      } else if (typeString.equals("Any")) {
        return InferredTypes.anyType();
      } else {
        // Try to make a qualified name. pytype does not give a qualified name for classes defined in the same file.
        // The filename should contain the module path, but we may get and extra prefix. The prefix will get removed later.
        var qualifiedTypeString = fileName.replace('/', '.').substring(0, fileName.lastIndexOf('.') + 1) + typeString;
        ClassSymbolImpl callableClassSymbol = new ClassSymbolImpl(typeString, qualifiedTypeString);
        return new RuntimeType(callableClassSymbol);
      }
  }

  private Optional<InferredType> getLocallyDefinedClassSymbolType(String typeString, String fileName, Tree tree) {
    fileName = fileName.substring(0, fileName.lastIndexOf('.'));
    Symbol ourSymbol = null;
    String symbolFullyQualifiedName = String.join(".", fileName, typeString);
    String symbolName = symbolFullyQualifiedName.substring(symbolFullyQualifiedName.lastIndexOf('.') + 1);
    while (ourSymbol == null && !tree.is(Tree.Kind.FILE_INPUT)) {
      tree = TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.FILE_INPUT);
      Scope currentScope = this.scopesByRootTree.get(tree);
      ourSymbol = Optional.ofNullable(currentScope.symbolsByName.get(symbolName))
        .orElse(resolveFqnSymbol(typeString, currentScope));
    }
    return Optional.ofNullable(ourSymbol).map(InferredTypes::runtimeType);
  }

  private Symbol resolveFqnSymbol(String symbolFullyQualifiedName, Scope currentScope) {
    String[] fqnSplit = symbolFullyQualifiedName.split("\\.");
    SymbolImpl currentSymbol = null;
    Optional<Symbol> classSymbol = Optional.empty();
    for (int i = 0; i < fqnSplit.length; i++) {
      currentSymbol = ((SymbolImpl) currentScope.symbolsByName.get(fqnSplit[i]));
      if (currentSymbol != null) {
        classSymbol = getImportedClassSymbol(currentSymbol, fqnSplit, i);
      }
      if (classSymbol.map(Symbol::fullyQualifiedName)
        .filter(symbolFullyQualifiedName::equals).isPresent()) {
        return classSymbol.get();
      }
    }
    return null;
  }

  private static Optional<Symbol> getImportedClassSymbol(SymbolImpl currentSymbol, String[] fqnSplit, int index) {
    while (index < fqnSplit.length) {
      if (index == fqnSplit.length - 1) {
        return Optional.of(currentSymbol);
      }
      index++;
      currentSymbol = ((SymbolImpl) currentSymbol.getChildrenSymbolByName().get(fqnSplit[index]));
      if (currentSymbol == null) {
        return Optional.empty();
      }
    }
    return Optional.of(currentSymbol);
  }

  private static String getBaseType(String typeString) { // Tuple[int, int]
    if (typeString.startsWith("Tuple")) {
      return "tuple";
    } else if (typeString.startsWith("List")) {
      return "list";
    } else if (typeString.startsWith("Set")) {
      return "set";
    } else if (typeString.startsWith("Dict")) {
      return "dict";
    } else if (typeString.startsWith("Type")) {
      return "type";
    } else {
      return typeString;
    }
  }

  public Optional<InferredType> getTypeFor(String fileName, Name name) {
    Token token = name.firstToken();
    return getTypeFor(fileName, token.line(), token.column(), name.name(), "Variable", name);
  }

  public Optional<InferredType> getTypeFor(String fileName, QualifiedExpression attributeAccess) {
    Token token = attributeAccess.firstToken();
    return getTypeFor(fileName, token.line(), token.column(), attributeAccess.name().name(), "Attribute", attributeAccess);
  }

  public PyTypeTable pyTypeTable() {
    return pyTypeTable;
  }
}
