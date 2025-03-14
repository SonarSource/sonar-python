/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.semantic.v2.types;

import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NoneExpression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.TypeTable;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.ComprehensionExpressionImpl;
import org.sonar.python.tree.DictCompExpressionImpl;
import org.sonar.python.tree.DictionaryLiteralImpl;
import org.sonar.python.tree.ListLiteralImpl;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.NoneExpressionImpl;
import org.sonar.python.tree.NumericLiteralImpl;
import org.sonar.python.tree.SetLiteralImpl;
import org.sonar.python.tree.StringLiteralImpl;
import org.sonar.python.tree.SubscriptionExpressionImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.types.v2.SpecialFormType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.types.v2.TypeChecker;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.tree.TreeUtils.locationInFile;

public class TrivialTypeInferenceVisitor extends BaseTreeVisitor {

  private final TypeTable projectLevelTypeTable;
  private final String fileId;
  private final String fullyQualifiedModuleName;
  private final String moduleName;

  private final Deque<Scope> typeStack = new ArrayDeque<>();
  private final Set<String> importedModulesFQN = new HashSet<>();
  private final TypeChecker typeChecker;
  private final List<String> typeVarNames = new ArrayList<>();

  private final Map<String, TypeWrapper> wildcardImportedTypes = new HashMap<>();

  private record Scope(PythonType type, String scopeFullyQualifiedName) {}

  public TrivialTypeInferenceVisitor(TypeTable projectLevelTypeTable, PythonFile pythonFile, String fullyQualifiedModuleName) {
    this.projectLevelTypeTable = projectLevelTypeTable;
    Path path = pathOf(pythonFile);
    this.moduleName = pythonFile.fileName();
    this.fileId = path != null ? path.toString() : pythonFile.toString();
    this.fullyQualifiedModuleName = fullyQualifiedModuleName;
    this.typeChecker = new TypeChecker(projectLevelTypeTable);
  }

  public Set<String> importedModulesFQN() {
    return importedModulesFQN;
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    var type = new ModuleType(moduleName);
    inTypeScope(type, () -> super.visitFileInput(fileInput));
  }

  @Override
  public void visitStringLiteral(StringLiteral stringLiteral) {
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    // TODO: SONARPY-1867 multiple object types to represent str instance?
    PythonType strType = builtins.resolveMember("str").orElse(PythonType.UNKNOWN);
    ((StringLiteralImpl) stringLiteral).typeV2(new ObjectType(strType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitTuple(Tuple tuple) {
    super.visitTuple(tuple);
    List<PythonType> contentTypes = tuple.elements().stream().map(Expression::typeV2).distinct().toList();
    List<PythonType> attributes = new ArrayList<>();
    if (contentTypes.size() == 1 && !contentTypes.get(0).equals(PythonType.UNKNOWN)) {
      attributes = contentTypes;
    }
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    PythonType tupleType = builtins.resolveMember("tuple").orElse(PythonType.UNKNOWN);
    ((TupleImpl) tuple).typeV2(new ObjectType(tupleType, attributes, new ArrayList<>()));
  }

  @Override
  public void visitDictionaryLiteral(DictionaryLiteral dictionaryLiteral) {
    super.visitDictionaryLiteral(dictionaryLiteral);
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    PythonType dictType = builtins.resolveMember("dict").orElse(PythonType.UNKNOWN);
    ((DictionaryLiteralImpl) dictionaryLiteral).typeV2(new ObjectType(dictType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitSetLiteral(SetLiteral setLiteral) {
    super.visitSetLiteral(setLiteral);
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    PythonType setType = builtins.resolveMember("set").orElse(PythonType.UNKNOWN);
    ((SetLiteralImpl) setLiteral).typeV2(new ObjectType(setType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitNumericLiteral(NumericLiteral numericLiteral) {
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    NumericLiteralImpl numericLiteralImpl = (NumericLiteralImpl) numericLiteral;
    NumericLiteralImpl.NumericKind numericKind = numericLiteralImpl.numericKind();
    PythonType pythonType = builtins.resolveMember(numericKind.value()).orElse(PythonType.UNKNOWN);
    numericLiteralImpl.typeV2(new ObjectType(pythonType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitNone(NoneExpression noneExpression) {
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    // TODO: SONARPY-1867 multiple object types to represent str instance?
    PythonType noneType = builtins.resolveMember("NoneType").orElse(PythonType.UNKNOWN);
    ((NoneExpressionImpl) noneExpression).typeV2(new ObjectType(noneType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitListLiteral(ListLiteral listLiteral) {
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    scan(listLiteral.elements());

    var candidateTypes = listLiteral.elements()
      .expressions()
      .stream()
      .map(Expression::typeV2)
      .distinct()
      .toList();

    var elementsType = UnionType.or(candidateTypes);

    var attributes = new ArrayList<PythonType>();
    attributes.add(elementsType);
    PythonType listType = builtins.resolveMember("list").orElse(PythonType.UNKNOWN);
    ((ListLiteralImpl) listLiteral).typeV2(new ObjectType(listType, attributes, new ArrayList<>()));
  }

  @Override
  public void visitPyListOrSetCompExpression(ComprehensionExpression comprehensionExpression) {
    super.visitPyListOrSetCompExpression(comprehensionExpression);
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    var pythonType = switch (comprehensionExpression.getKind()) {
      case LIST_COMPREHENSION -> builtins.resolveMember("list").orElse(PythonType.UNKNOWN);
      case SET_COMPREHENSION -> builtins.resolveMember("set").orElse(PythonType.UNKNOWN);
      default -> PythonType.UNKNOWN;
    };
    ((ComprehensionExpressionImpl) comprehensionExpression).typeV2(new ObjectType(pythonType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitDictCompExpression(DictCompExpression dictCompExpression) {
    super.visitDictCompExpression(dictCompExpression);
    var builtins = this.projectLevelTypeTable.getBuiltinsModule();
    var dictType = builtins.resolveMember("dict").orElse(PythonType.UNKNOWN);
    ((DictCompExpressionImpl) dictCompExpression).typeV2(new ObjectType(dictType, new ArrayList<>(), new ArrayList<>()));
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    scan(classDef.args());
    Name name = classDef.name();
    ClassType type = buildClassType(classDef);
    ((NameImpl) name).typeV2(type);

    inTypeScope(type, () -> scan(classDef.body()));
  }

  private ClassType buildClassType(ClassDef classDef) {
    Name className = classDef.name();
    String fullyQualifiedName = currentScopeFullyQualifiedName() + "." + classDef.name().name();
    ClassTypeBuilder classTypeBuilder = new ClassTypeBuilder(className.name(), fullyQualifiedName)
      .withHasDecorators(!classDef.decorators().isEmpty())
      .withDefinitionLocation(locationInFile(className, fileId));
    resolveTypeHierarchy(classDef, classTypeBuilder);
    if (classDef.typeParams() != null) {
      classTypeBuilder.withIsGeneric(true);
    }
    ClassType classType = classTypeBuilder.build();

    if (currentType() instanceof ClassType ownerClass) {
      SymbolV2 symbolV2 = className.symbolV2();
      if (symbolV2 != null) {
        PythonType memberType = symbolV2.hasSingleBindingUsage() ? classType : PythonType.UNKNOWN;
        ownerClass.members().add(new Member(classType.name(), memberType));
      }
    }
    return classType;
  }

  void resolveTypeHierarchy(ClassDef classDef, ClassTypeBuilder classTypeBuilder) {
    Optional.of(classDef)
      .map(ClassDef::args)
      .map(ArgList::arguments)
      .stream()
      .flatMap(Collection::stream)
      .forEach(argument -> {
        if (argument instanceof RegularArgument regularArgument) {
          this.addParentClass(classTypeBuilder, regularArgument);
        } else {
          classTypeBuilder.addSuperClass(PythonType.UNKNOWN);
        }
      });
  }

  private void addParentClass(ClassTypeBuilder classTypeBuilder, RegularArgument regularArgument) {
    Name keyword = regularArgument.keywordArgument();
    // TODO: SONARPY-1871 store names if not resolved properly
    if (keyword != null) {
      if ("metaclass".equals(keyword.name())) {
        PythonType argumentType = getTypeV2FromArgument(regularArgument);
        classTypeBuilder.metaClasses().add(argumentType);
      }
      return;
    }
    PythonType argumentType = getTypeV2FromArgument(regularArgument);
    classTypeBuilder.addSuperClass(argumentType);
    if (isParentAGenericClass(regularArgument, argumentType)) {
      classTypeBuilder.withIsGeneric(true);
    }
  }

  private boolean isParentAGenericClass(RegularArgument regularArgument, PythonType argumentType) {
    return typeChecker.typeCheckBuilder().isGeneric().check(argumentType) == TriBool.TRUE
      && regularArgument.expression() instanceof SubscriptionExpression subscriptionExpression
      && subscriptionExpression.subscripts().expressions().stream().anyMatch(expression -> expression instanceof Name name && typeVarNames.contains(name.name()));
  }

  private static PythonType getTypeV2FromArgument(RegularArgument regularArgument) {
    Expression expression = regularArgument.expression();
    // Ensure we support correctly typing symbols like "List[str] / list[str]"
    return expression.typeV2();
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    scan(functionDef.decorators());
    scan(functionDef.typeParams());
    scan(functionDef.parameters());
    scan(functionDef.returnTypeAnnotation());
    FunctionType functionType = buildFunctionType(functionDef);
    ((NameImpl) functionDef.name()).typeV2(functionType);
    inTypeScope(functionType, () -> {
      // TODO: check scope accuracy
      scan(functionDef.typeParams());
      scan(functionDef.parameters());
      scan(functionDef.body());
    });
  }

  private FunctionType buildFunctionType(FunctionDef functionDef) {
    String fullyQualifiedName = currentScopeFullyQualifiedName() + "." + functionDef.name().name();
    FunctionTypeBuilder functionTypeBuilder = new FunctionTypeBuilder()
      .fromFunctionDef(functionDef, fullyQualifiedName, fileId, projectLevelTypeTable)
      .withDefinitionLocation(locationInFile(functionDef.name(), fileId));
    ClassType owner = null;
    if (currentType() instanceof ClassType classType) {
      owner = classType;
    }
    if (owner != null) {
      functionTypeBuilder.withOwner(owner);
    }
    TypeAnnotation typeAnnotation = functionDef.returnTypeAnnotation();
    if (typeAnnotation != null) {
      PythonType returnType = typeAnnotation.expression().typeV2();
      functionTypeBuilder.withReturnType(returnType instanceof UnknownType ? returnType : new ObjectType(returnType, TypeSource.TYPE_HINT));
      functionTypeBuilder.withTypeOrigin(TypeOrigin.LOCAL);
    }
    FunctionType functionType = functionTypeBuilder.build();
    SymbolV2 symbolV2 = functionDef.name().symbolV2();
    if (owner != null && symbolV2 != null) {
      if (symbolV2.hasSingleBindingUsage()) {
        owner.members().add(new Member(functionType.name(), functionType));
      } else {
        owner.members().add(new Member(functionType.name(), PythonType.UNKNOWN));
      }
    }
    return functionType;
  }

  @Override
  public void visitImportName(ImportName importName) {
    importName.modules()
      .forEach(aliasedName -> {
        var dottedName = aliasedName.dottedName();
        var fqn = dottedNameToPartFqn(dottedName);
        var resolvedType = projectLevelTypeTable.getModuleType(fqn);
        importedModulesFQN.add(String.join(".", fqn));
        if (aliasedName.alias() != null) {
          generateNamesForImportAlias(aliasedName, resolvedType, fqn);
        } else {
          generateNames(resolvedType, dottedName.names(), fqn);
        }
      });
  }

  private static void generateNamesForImportAlias(AliasedName aliasedName, PythonType resolvedType, List<String> fqn) {
    var aliasedNameType = resolvedType instanceof UnknownType ? new UnknownType.UnresolvedImportType(String.join(".", fqn)) : resolvedType;
    setTypeToName(aliasedName.alias(), aliasedNameType);
  }

  private static void generateNames(PythonType resolvedType, List<Name> names, List<String> fqn) {
    if (resolvedType instanceof ModuleType module) {
      for (int i = names.size() - 1; i >= 0; i--) {
        setTypeToName(names.get(i), module);
        module = Optional.ofNullable(module)
          .map(ModuleType::parent)
          .orElse(null);
      }
    } else if (resolvedType instanceof UnknownType) {
      for (int i = names.size() - 1; i >= 0; i--) {
        UnknownType.UnresolvedImportType type = new UnknownType.UnresolvedImportType(String.join(".", fqn.subList(0, i + 1)));
        setTypeToName(names.get(i), type);
      }
    }
  }

  @Override
  public void visitImportFrom(ImportFrom importFrom) {
    List<String> fromModuleFqn = Optional.ofNullable(importFrom.module())
      .map(TrivialTypeInferenceVisitor::dottedNameToPartFqn)
      .orElse(new ArrayList<>());
    List<Token> dotPrefixTokens = importFrom.dottedPrefixForModule();
    if (!dotPrefixTokens.isEmpty()) {
      // Relative import: we start from the current module FQN and go up as many levels as there are dots in the import statement
      List<String> moduleFqnElements = List.of(fullyQualifiedModuleName.split("\\."));
      int sizeLimit = Math.max(0, moduleFqnElements.size() - dotPrefixTokens.size());
      fromModuleFqn = Stream.concat(moduleFqnElements.stream().limit(sizeLimit), fromModuleFqn.stream()).toList();
    }
    importedModulesFQN.add(String.join(".", fromModuleFqn));
    setTypeToImportFromStatement(importFrom, fromModuleFqn);

    if(importFrom.isWildcardImport()) {
      collectWildcardImportedSymbols(fromModuleFqn);
    }
  }

  private void collectWildcardImportedSymbols(List<String> moduleFqn) {
    PythonType resolvedModuleType = projectLevelTypeTable.getModuleType(moduleFqn);
    if(resolvedModuleType instanceof ModuleType moduleType) {
      wildcardImportedTypes.putAll(moduleType.members());
    }
  }

  private static List<String> dottedNameToPartFqn(DottedName dottedName) {
    return dottedName.names()
      .stream()
      .map(Name::name)
      .toList();
  }

  private void setTypeToImportFromStatement(ImportFrom importFrom, List<String> fqn) {
    var module = projectLevelTypeTable.getModuleType(fqn);
    for (var aliasedName : importFrom.importedNames()) {
      aliasedName.dottedName().names()
        .stream()
        .findFirst()
        .ifPresent(name -> {
          var type = module.resolveMember(name.name()).orElseGet(() -> createUnresolvedImportType(fqn, name));

          var boundName = Optional.ofNullable(aliasedName.alias())
            .orElse(name);

          setTypeToName(boundName, type);
        });
    }
  }

  private static UnknownType.UnresolvedImportType createUnresolvedImportType(List<String> moduleFqnList, Name name) {
    String fromModuleFqn = String.join(".", moduleFqnList);
    String fqn = fromModuleFqn.isEmpty() ? name.name() : String.join(".", fromModuleFqn, name.name());
    return new UnknownType.UnresolvedImportType(fqn);
  }

  @Override
  public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
    scan(assignmentStatement.assignedValue());
    scan(assignmentStatement.lhsExpressions());

    getFirstAssignmentName(assignmentStatement).ifPresent(lhsName -> {
      var assignedValueType = assignmentStatement.assignedValue().typeV2();
      lhsName.typeV2(assignedValueType);
      addStaticFieldToClass(lhsName, PythonType.UNKNOWN);
    });
  }

  @Override
  public void visitAnnotatedAssignment(AnnotatedAssignment assignmentStatement) {
    super.visitAnnotatedAssignment(assignmentStatement);
    Optional.ofNullable(assignmentStatement.variable())
      .filter(NameImpl.class::isInstance)
      .map(NameImpl.class::cast)
      .ifPresent(lhsName -> {
        if (currentType() instanceof ClassType) {
          Optional.ofNullable(assignmentStatement.annotation())
            .map(TypeAnnotation::expression)
            .map(Expression::typeV2)
            .filter(Predicate.not(PythonType.UNKNOWN::equals))
            .map(t -> new ObjectType(t, TypeSource.TYPE_HINT))
            .ifPresent(lhsName::typeV2);
          addStaticFieldToClass(lhsName, lhsName.typeV2());
        }
      });
  }

  @Override
  public void visitCallExpression(CallExpression callExpression) {
    super.visitCallExpression(callExpression);
    assignPossibleTypeVar(callExpression);
  }

  private void assignPossibleTypeVar(CallExpression callExpression) {
    PythonType pythonType = callExpression.callee().typeV2();
    TriBool check = typeChecker.typeCheckBuilder().isTypeWithName("typing.TypeVar").check(pythonType);
    if (check == TriBool.TRUE) {
      Tree parent = TreeUtils.firstAncestor(callExpression, t -> t.is(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT, Tree.Kind.ASSIGNMENT_EXPRESSION));
      Optional<Name> assignedName = Optional.empty();
      if (parent instanceof AssignmentStatement assignmentStatement) {
        assignedName = extractAssignedName(assignmentStatement);
      }
      if (parent instanceof AnnotatedAssignment annotatedAssignment) {
        assignedName = extractAssignedName(annotatedAssignment);
      }
      if (parent instanceof AssignmentExpression assignmentExpression) {
        assignedName = extractAssignedName(assignmentExpression);
      }
      assignedName.ifPresent(name -> typeVarNames.add(name.name()));
    }
  }

  private static Optional<Name> extractAssignedName(AssignmentExpression assignmentExpression) {
    return Optional.of(assignmentExpression.lhsName()).map(Name.class::cast);
  }

  private static Optional<Name> extractAssignedName(AnnotatedAssignment annotatedAssignment) {
    return Optional.of(annotatedAssignment.variable()).filter(v -> v.is(Tree.Kind.NAME)).map(Name.class::cast);
  }

  private static Optional<Name> extractAssignedName(AssignmentStatement assignmentStatement) {
    return assignmentStatement.lhsExpressions().stream().findFirst()
      .map(ExpressionList::expressions).stream().flatMap(List::stream).findFirst().filter(v -> v.is(Tree.Kind.NAME)).map(Name.class::cast);
  }

  private static Optional<NameImpl> getFirstAssignmentName(AssignmentStatement assignmentStatement) {
    return Optional.of(assignmentStatement)
      .map(AssignmentStatement::lhsExpressions)
      .filter(lhs -> lhs.size() == 1)
      .map(lhs -> lhs.get(0))
      .map(ExpressionList::expressions)
      .filter(lhs -> lhs.size() == 1)
      .map(lhs -> lhs.get(0))
      .filter(NameImpl.class::isInstance)
      .map(NameImpl.class::cast);
  }

  private void addStaticFieldToClass(Name name, PythonType type) {
    if (currentType() instanceof ClassType ownerClass) {
      var memberName = name.name();
      var toRemove = ownerClass.members().stream().filter(member -> memberName.equals(member.name())).toList();
      ownerClass.members().removeAll(toRemove);
      ownerClass.members().add(new Member(memberName, type));
    }
  }

  @Override
  public void visitParameter(Parameter parameter) {
    scan(parameter.typeAnnotation());
    scan(parameter.defaultValue());
    Optional.ofNullable(parameter.typeAnnotation())
      .map(TypeAnnotation::expression)
      .map(TrivialTypeInferenceVisitor::resolveTypeAnnotationExpressionType)
      .ifPresent(type -> setTypeToName(parameter.name(), type));
    scan(parameter.name());
  }

  private static PythonType resolveTypeAnnotationExpressionType(Expression expression) {
    if (expression instanceof Name name && name.typeV2() != PythonType.UNKNOWN) {
      return new ObjectType(name.typeV2(), TypeSource.TYPE_HINT);
    } else if (expression instanceof SubscriptionExpression subscriptionExpression && subscriptionExpression.object().typeV2() != PythonType.UNKNOWN) {
      var candidateTypes = subscriptionExpression.subscripts()
        .expressions()
        .stream()
        .map(Expression::typeV2)
        .distinct()
        .toList();

      var elementsType = UnionType.or(candidateTypes);

      var attributes = new ArrayList<PythonType>();
      attributes.add(new ObjectType(elementsType, TypeSource.TYPE_HINT));
      return new ObjectType(subscriptionExpression.object().typeV2(), attributes, new ArrayList<>(), TypeSource.TYPE_HINT);
    } else if (expression instanceof BinaryExpression binaryExpression) {
      var left = resolveTypeAnnotationExpressionType(binaryExpression.leftOperand());
      var right = resolveTypeAnnotationExpressionType(binaryExpression.rightOperand());
      // TODO: we need to make a decision on should here be a union type of object types or an object type of a union type.
      //  ATM it is blocked by the generic types resolution redesign
      return UnionType.or(left, right);
    } else if (expression.typeV2() instanceof ClassType classType) {
      return new ObjectType(classType, TypeSource.TYPE_HINT);
    }
    return PythonType.UNKNOWN;
  }

  @Override
  public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
    scan(qualifiedExpression.qualifier());
    if (qualifiedExpression.name() instanceof NameImpl name) {
      var nameType = qualifiedExpression.qualifier().typeV2()
        .resolveMember(qualifiedExpression.name().name())
        .orElse(PythonType.UNKNOWN);
      name.typeV2(nameType);
    }
  }

  @Override
  public void visitSubscriptionExpression(SubscriptionExpression subscriptionExpression) {
    super.visitSubscriptionExpression(subscriptionExpression);
    PythonType pythonType = subscriptionExpression.object().typeV2();
    if (typeChecker.typeCheckBuilder().isGeneric().check(pythonType) == TriBool.TRUE) {
      ((SubscriptionExpressionImpl) subscriptionExpression).typeV2(pythonType);
    }
  }

  @Override
  public void visitName(Name name) {
    SymbolV2 symbolV2 = name.symbolV2();
    if (symbolV2 == null) {
//    This part could be affected by SONARPY-1802
      var builtInType = projectLevelTypeTable.getBuiltinsModule().resolveMember(name.name());
      if (builtInType.isPresent()) {
        setTypeToName(name, builtInType.get());
      } else {
        TypeWrapper wildcardImportedType = wildcardImportedTypes.get(name.name());
        if (wildcardImportedType != null) {
          setTypeToName(name, wildcardImportedType.type());
        }
      }
      return;
    }

    var bindingUsages = new ArrayList<UsageV2>();
    for (var usage : symbolV2.usages()) {
      if (usage.kind().equals(UsageV2.Kind.GLOBAL_DECLARATION)) {
        // Don't infer type for global variables
        return;
      }
      if (usage.isBindingUsage()) {
        bindingUsages.add(usage);
      }
      if (bindingUsages.size() > 1) {
        // no need to iterate over usages if there is more than one binding usage
        return;
      }
    }

    bindingUsages.stream()
      .findFirst()
      .filter(UsageV2::isBindingUsage)
      .map(UsageV2::tree)
      .filter(Expression.class::isInstance)
      .map(Expression.class::cast)
      .map(Expression::typeV2)
      // TODO: classes (SONARPY-1829) and functions should be propagated like other types
      .filter(TrivialTypeInferenceVisitor::shouldTypeBeEagerlyPropagated)
      .ifPresent(type -> setTypeToName(name, type));

  }

  private static boolean shouldTypeBeEagerlyPropagated(PythonType t) {
    return (t instanceof ClassType)
           || (t instanceof FunctionType)
           || (t instanceof ModuleType)
           || (t instanceof UnknownType.UnresolvedImportType)
           || (t instanceof SpecialFormType);
  }

  private PythonType currentType() {
    return typeStack.peek().type();
  }

  private String currentScopeFullyQualifiedName() {
    return typeStack.peek().scopeFullyQualifiedName();
  }

  private void inTypeScope(PythonType pythonType, Runnable runnable) {
    String newScopeFullyQualifiedName = Optional.ofNullable(typeStack.peek())
      .map(Scope::scopeFullyQualifiedName)
      .map(s -> s + "." + pythonType.name())
      .orElse(fullyQualifiedModuleName);
    this.typeStack.push(new Scope(pythonType, newScopeFullyQualifiedName));
    runnable.run();
    this.typeStack.poll();
  }

  private static void setTypeToName(@Nullable Tree tree, @Nullable PythonType type) {
    if (tree instanceof NameImpl name && type != null) {
      name.typeV2(type);
    }
  }
}
