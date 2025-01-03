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
package org.sonar.python.semantic;


import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.sonar.python.semantic.SymbolUtils.isPrivateName;
import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.tree.TreeUtils.locationInFile;
import static org.sonar.python.types.TypeShed.isValidForProjectPythonVersion;
import static org.sonar.python.types.TypeShed.symbolsFromProtobufDescriptors;

public class ClassSymbolImpl extends SymbolImpl implements ClassSymbol {

  private final List<Symbol> superClasses = new ArrayList<>();
  private List<String> superClassesFqns = new ArrayList<>();
  private List<String> inlinedSuperClassFqn = new ArrayList<>();
  private Set<Symbol> allSuperClasses = null;
  private Set<Symbol> allSuperClassesIncludingAmbiguousSymbols = null;
  private boolean hasSuperClassWithoutSymbol = false;
  private final Set<Symbol> members = new HashSet<>();
  private Map<String, Symbol> membersByName = null;
  private boolean hasAlreadyReadSuperClasses = false;
  private boolean hasAlreadyReadMembers = false;
  private boolean hasDecorators = false;
  private boolean hasMetaClass = false;
  private final LocationInFile classDefinitionLocation;
  @Nullable
  private String metaclassFQN = null;
  private boolean supportsGenerics = false;

  public ClassSymbolImpl(ClassDef classDef, @Nullable String fullyQualifiedName, PythonFile pythonFile) {
    super(classDef.name().name(), fullyQualifiedName);
    this.setKind(Kind.CLASS);
    Path path = pathOf(pythonFile);
    String fileId = path != null ? path.toString() : pythonFile.toString();
    hasDecorators = !classDef.decorators().isEmpty();
    classDefinitionLocation = locationInFile(classDef.name(), fileId);
  }

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    super(name, fullyQualifiedName);
    classDefinitionLocation = null;
    hasDecorators = false;
    hasMetaClass = false;
    metaclassFQN = null;
    supportsGenerics = false;
    setKind(Kind.CLASS);
  }

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName, LocationInFile location) {
    super(name, fullyQualifiedName);
    classDefinitionLocation = location;
    hasDecorators = false;
    hasMetaClass = false;
    metaclassFQN = null;
    supportsGenerics = false;
    setKind(Kind.CLASS);
  }

  public ClassSymbolImpl(ClassDescriptor classDescriptor, String symbolName) {
    super(symbolName, classDescriptor.fullyQualifiedName());
    setKind(Kind.CLASS);
    classDefinitionLocation = classDescriptor.definitionLocation();
    hasDecorators = classDescriptor.hasDecorators();
    hasMetaClass = classDescriptor.hasMetaClass();
    metaclassFQN = classDescriptor.metaclassFQN();
    supportsGenerics = classDescriptor.supportsGenerics();
    hasSuperClassWithoutSymbol = classDescriptor.hasSuperClassWithoutDescriptor();
  }

  public static ClassSymbol copyFrom(String name, ClassSymbol classSymbol) {
    return new ClassSymbolImpl(name, classSymbol);
  }

  private ClassSymbolImpl(String name, ClassSymbol classSymbol) {
    super(name, classSymbol.fullyQualifiedName());
    classDefinitionLocation = classSymbol.definitionLocation();
    hasDecorators = classSymbol.hasDecorators();
    hasMetaClass = ((ClassSymbolImpl) classSymbol).hasMetaClass();
    metaclassFQN = ((ClassSymbolImpl) classSymbol).metaclassFQN;
    supportsGenerics = ((ClassSymbolImpl) classSymbol).supportsGenerics;
    validForPythonVersions = ((ClassSymbolImpl) classSymbol).validForPythonVersions;
    superClassesFqns = ((ClassSymbolImpl) classSymbol).superClassesFqns;
    setKind(Kind.CLASS);
  }

  public ClassSymbolImpl(SymbolsProtos.ClassSymbol classSymbolProto, String moduleName) {
    super(classSymbolProto.getName(), TypeShed.normalizedFqn(classSymbolProto.getFullyQualifiedName(), moduleName, classSymbolProto.getName()));
    setKind(Kind.CLASS);
    classDefinitionLocation = null;
    hasDecorators = classSymbolProto.getHasDecorators();
    hasMetaClass = classSymbolProto.getHasMetaclass();
    metaclassFQN = classSymbolProto.getMetaclassName();
    supportsGenerics = classSymbolProto.getIsGeneric();
    Set<Symbol> classMembers = new HashSet<>();
    Map<String, Set<Object>> descriptorsByFqn = new HashMap<>();
    classSymbolProto.getMethodsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByFqn.computeIfAbsent(proto.getFullyQualifiedName(), d -> new HashSet<>()).add(proto));
    classSymbolProto.getOverloadedMethodsList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByFqn.computeIfAbsent(proto.getFullname(), d -> new HashSet<>()).add(proto));
    classSymbolProto.getAttributesList().stream()
      .filter(d -> isValidForProjectPythonVersion(d.getValidForList()))
      .forEach(proto -> descriptorsByFqn.computeIfAbsent(proto.getFullyQualifiedName(), d -> new HashSet<>()).add(proto));

    inlineInheritedMethodsFromPrivateClass(classSymbolProto.getSuperClassesList(), descriptorsByFqn);

    for (Map.Entry<String, Set<Object>> entry : descriptorsByFqn.entrySet()) {
      Set<Symbol> symbols = symbolsFromProtobufDescriptors(entry.getValue(), fullyQualifiedName, moduleName, true);
      classMembers.add(symbols.size() > 1 ? AmbiguousSymbolImpl.create(symbols) : symbols.iterator().next());
    }
    addMembers(classMembers);
    superClassesFqns.addAll(classSymbolProto.getSuperClassesList().stream().map(TypeShed::normalizedFqn).toList());
    superClassesFqns.removeAll(inlinedSuperClassFqn);
    validForPythonVersions = new HashSet<>(classSymbolProto.getValidForList());
  }

  private void inlineInheritedMethodsFromPrivateClass(List<String> superClassesFqns, Map<String, Set<Object>> descriptorsByFqn) {
    for (String superClassFqn : superClassesFqns) {
      if (isPrivateName(superClassFqn)) {
        SymbolsProtos.ClassSymbol superClass = TypeShed.classDescriptorWithFQN(superClassFqn);
        if (superClass == null) return;
        inlinedSuperClassFqn.add(superClassFqn);
        for (SymbolsProtos.FunctionSymbol functionSymbol : superClass.getMethodsList()) {
          String methodFqn = this.fullyQualifiedName + "." + functionSymbol.getName();
          descriptorsByFqn.computeIfAbsent(methodFqn, d -> new HashSet<>()).add(functionSymbol);
        }
        for (SymbolsProtos.OverloadedFunctionSymbol functionSymbol : superClass.getOverloadedMethodsList()) {
          String methodFqn = this.fullyQualifiedName + "." + functionSymbol.getName();
          descriptorsByFqn.computeIfAbsent(methodFqn, d -> new HashSet<>()).add(functionSymbol);
        }
        this.superClassesFqns.addAll(superClass.getSuperClassesList());
      }
    }
  }

  @Override
  public ClassSymbolImpl copyWithoutUsages() {
    ClassSymbolImpl copiedClassSymbol = new ClassSymbolImpl(name(), this);
    if (hasEvaluatedSuperClasses()) {
      for (Symbol superClass : superClasses()) {
        if (superClass == this) {
          copiedClassSymbol.superClasses.add(copiedClassSymbol);
        } else if (superClass.is(Kind.CLASS, Kind.AMBIGUOUS)) {
          copiedClassSymbol.superClasses.add(((SymbolImpl) superClass).copyWithoutUsages());
        } else {
          copiedClassSymbol.superClasses.add(new SymbolImpl(superClass.name(), superClass.fullyQualifiedName()));
        }
      }
    }
    copiedClassSymbol.addMembers(members.stream().map(m -> ((SymbolImpl) m).copyWithoutUsages()).collect(Collectors.toList()));
    if (hasSuperClassWithoutSymbol) {
      copiedClassSymbol.setHasSuperClassWithoutSymbol();
    }
    return copiedClassSymbol;
  }

  @Override
  public List<String> superClassesFqn() {
    return superClassesFqns;
  }

  public boolean shouldSearchHierarchyInTypeshed() {
    return !hasAlreadyReadSuperClasses && superClasses.isEmpty() && !superClassesFqns.isEmpty();
  }

  @Override
  public List<Symbol> superClasses() {
    // In case of symbols coming from TypeShed protobuf, we resolve superclasses lazily
    if (!hasAlreadyReadSuperClasses && superClasses.isEmpty() && !superClassesFqns.isEmpty()) {
      superClassesFqns.stream().map(SymbolUtils::typeshedSymbolWithFQN).forEach(this::addSuperClass);
    }
    hasAlreadyReadSuperClasses = true;
    return Collections.unmodifiableList(superClasses);
  }

  public void addSuperClass(Symbol symbol) {
    if (hasAlreadyReadSuperClasses) {
      throw new IllegalStateException("Cannot call addSuperClass, super classes were already read");
    }
    this.superClasses.add(symbol);
  }

  @Override
  public boolean hasUnresolvedTypeHierarchy() {
    return hasUnresolvedTypeHierarchy(true);
  }

  public boolean hasUnresolvedTypeHierarchy(boolean includeAmbiguousSymbols) {
    for (Symbol superClassSymbol : allSuperClasses(includeAmbiguousSymbols)) {
      if (superClassSymbol.kind() != Kind.CLASS) {
        return true;
      }

      ClassSymbolImpl superClass = (ClassSymbolImpl) superClassSymbol;
      if (superClass.hasSuperClassWithoutSymbol) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Set<Symbol> declaredMembers() {
    hasAlreadyReadMembers = true;
    return members;
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    for (Symbol symbol : allSuperClasses(false)) {
      if (symbol.kind() == Kind.CLASS) {
        ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
        Symbol matchingMember = classSymbol.membersByName().get(memberName);
        if (matchingMember != null) {
          return Optional.of(matchingMember);
        }
      }
    }
    return Optional.empty();
  }

  @Override
  public boolean hasMetaClass() {
    return hasMetaClass || membersByName().get("__metaclass__") != null;
  }

  @Override
  public boolean canHaveMember(String memberName) {
    if (hasUnresolvedTypeHierarchy() || hasSuperClassWithUnknownMetaClass()) {
      return true;
    }
    for (Symbol symbol : allSuperClasses(true)) {
      if (symbol.kind() == Kind.CLASS) {
        ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
        Symbol matchingMember = classSymbol.membersByName().get(memberName);
        if (matchingMember != null) {
          return true;
        }
      }
    }
    return false;
  }

  public boolean hasSuperClassWithUnknownMetaClass() {
    for (Symbol symbol : allSuperClasses(true)) {
      if (symbol.is(Kind.CLASS)) {
        ClassSymbolImpl superClass = (ClassSymbolImpl) symbol;
        // excluding ABCMeta because it doesn't add extra methods and to avoid FN for typeshed symbols
        if (superClass.hasMetaClass() && !"abc.ABCMeta".equals(superClass.metaclassFQN())) {
          return true;
        }
      }
    }
    return false;
  }

  @Override
  public LocationInFile definitionLocation() {
    return classDefinitionLocation;
  }

  @Override
  public boolean isOrExtends(String fullyQualifiedClassName) {
    return allSuperClasses(false).stream().anyMatch(c -> c.fullyQualifiedName() != null && Objects.equals(fullyQualifiedClassName, c.fullyQualifiedName()));
  }

  @Override
  public boolean isOrExtends(ClassSymbol other) {
    if ("object".equals(other.fullyQualifiedName())) {
      return true;
    }
    // TODO there should be only 1 class with a given fullyQualifiedName when analyzing a python file
    return allSuperClasses(false).stream().anyMatch(c -> Objects.equals(c.fullyQualifiedName(), other.fullyQualifiedName()));
  }

  @Override
  public boolean canBeOrExtend(String fullyQualifiedClassName) {
    if ("object".equals(fullyQualifiedClassName)) {
      return true;
    }
    return allSuperClasses(true).stream().anyMatch(c -> c.fullyQualifiedName() != null && Objects.equals(fullyQualifiedClassName, c.fullyQualifiedName()))
      || hasUnresolvedTypeHierarchy();
  }

  @Override
  public boolean hasDecorators() {
    return hasDecorators;
  }

  private Map<String, Symbol> membersByName() {
    if (membersByName == null) {
      membersByName = declaredMembers().stream().collect(Collectors.toMap(Symbol::name, m -> m, (s1, s2) -> s1));
    }
    return membersByName;
  }

  public void addMembers(Collection<Symbol> members) {
    if (hasAlreadyReadMembers) {
      throw new IllegalStateException("Cannot call addMembers, members were already read");
    }
    this.members.addAll(members);
    members.stream()
      .filter(m -> m.kind() == Kind.FUNCTION)
      .forEach(m -> ((FunctionSymbolImpl) m).setOwner(this));
  }

  public void setHasSuperClassWithoutSymbol() {
    this.hasSuperClassWithoutSymbol = true;
  }

  public void setHasMetaClass() {
    this.hasMetaClass = true;
  }

  public void setMetaclassFQN(String metaclassFQN) {
    this.metaclassFQN = metaclassFQN;
  }

  @CheckForNull
  public String metaclassFQN() {
    return metaclassFQN;
  }

  private Set<Symbol> allSuperClasses(boolean includeAmbiguousSymbols) {
    if (!includeAmbiguousSymbols) {
      if (allSuperClasses == null) {
        allSuperClasses = new LinkedHashSet<>();
        exploreSuperClasses(this, allSuperClasses, false);
      }
      return allSuperClasses;
    }
    if (allSuperClassesIncludingAmbiguousSymbols == null) {
      allSuperClassesIncludingAmbiguousSymbols = new LinkedHashSet<>();
      exploreSuperClasses(this, allSuperClassesIncludingAmbiguousSymbols, true);
    }
    return allSuperClassesIncludingAmbiguousSymbols;
  }

  private static void exploreSuperClasses(Symbol symbol, Set<Symbol> set, boolean includeAmbiguousSymbols) {
    if (symbol.is(Kind.AMBIGUOUS) && includeAmbiguousSymbols) {
      AmbiguousSymbol ambiguousSymbol = (AmbiguousSymbol) symbol;
      for (Symbol alternative : ambiguousSymbol.alternatives()) {
        exploreSuperClasses(alternative, set, true);
      }
    } else if (set.add(symbol) && symbol.is(Kind.CLASS)) {
      ClassSymbol classSymbol = (ClassSymbol) symbol;
      for (Symbol superClass : classSymbol.superClasses()) {
        exploreSuperClasses(superClass, set, includeAmbiguousSymbols);
      }
    }
  }

  @Override
  public void removeUsages() {
    super.removeUsages();
    superClasses.forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
    members.forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
  }

  public boolean hasSuperClassWithoutSymbol() {
    return hasSuperClassWithoutSymbol;
  }

  public boolean supportsGenerics() {
    return supportsGenerics;
  }

  public void setSupportsGenerics(boolean supportsGenerics) {
    this.supportsGenerics = supportsGenerics;
  }

  /**
   * Precomputed typeshed class symbols might be "lazily evaluated", i.e. only information about super classes fqn is stored, without having created the actual
   * type hierarchy.
   * This method is used to know if super classes have been already created and added to {@link #superClasses}.
   * This might happen in the following cases:
   * - Super classes have been already read, hence class symbol is not lazy anymore
   * - {@link #superClassesFqns} is empty, meaning either this isn't a precomputed typeshed symbol or the class have no superclass.
   */
  public boolean hasEvaluatedSuperClasses() {
    return hasAlreadyReadSuperClasses || superClassesFqns.isEmpty();
  }
}
