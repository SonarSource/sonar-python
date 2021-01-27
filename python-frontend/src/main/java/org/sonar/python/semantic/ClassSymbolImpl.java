/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.semantic;


import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
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

import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.tree.TreeUtils.locationInFile;

public class ClassSymbolImpl extends SymbolImpl implements ClassSymbol {

  private final List<Symbol> superClasses = new ArrayList<>();
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
    String fileId = null;
    if (!SymbolUtils.isTypeShedFile(pythonFile)) {
      Path path = pathOf(pythonFile);
      fileId = path != null ? path.toString() : pythonFile.toString();
    }
    hasDecorators = !classDef.decorators().isEmpty();
    classDefinitionLocation = locationInFile(classDef.name(), fileId);
  }

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName) {
    this(name, fullyQualifiedName, null, false, false, null, false);
  }

  public ClassSymbolImpl(String name, @Nullable String fullyQualifiedName, @Nullable LocationInFile definitionLocation,
                         boolean hasDecorators, boolean hasMetaClass, @Nullable String metaclassFQN, boolean supportsGenerics) {
    super(name, fullyQualifiedName);
    classDefinitionLocation = definitionLocation;
    this.hasDecorators = hasDecorators;
    this.hasMetaClass = hasMetaClass;
    this.metaclassFQN = metaclassFQN;
    this.supportsGenerics = supportsGenerics;
    setKind(Kind.CLASS);
  }

  @Override
  ClassSymbolImpl copyWithoutUsages() {
    ClassSymbolImpl copiedClassSymbol = new ClassSymbolImpl(name(), fullyQualifiedName(), definitionLocation(), hasDecorators, hasMetaClass, metaclassFQN, supportsGenerics);
    for (Symbol superClass : superClasses()) {
      if (superClass == this) {
        copiedClassSymbol.superClasses.add(copiedClassSymbol);
      } else if (superClass.kind() == Symbol.Kind.CLASS) {
        copiedClassSymbol.superClasses.add(((ClassSymbolImpl) superClass).copyWithoutUsages());
      } else if (superClass.is(Kind.AMBIGUOUS)) {
        copiedClassSymbol.superClasses.add(((AmbiguousSymbolImpl) superClass).copyWithoutUsages());
      } else {
        copiedClassSymbol.superClasses.add(new SymbolImpl(superClass.name(), superClass.fullyQualifiedName()));
      }
    }
    copiedClassSymbol.addMembers(members.stream().map(m -> ((SymbolImpl) m).copyWithoutUsages()).collect(Collectors.toList()));
    if (hasSuperClassWithoutSymbol) {
      copiedClassSymbol.setHasSuperClassWithoutSymbol();
    }
    return copiedClassSymbol;
  }

  @Override
  public List<Symbol> superClasses() {
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

  boolean hasSuperClassWithoutSymbol() {
    return hasSuperClassWithoutSymbol;
  }

  public boolean supportsGenerics() {
    return supportsGenerics;
  }

  public void setSupportsGenerics(boolean supportsGenerics) {
    this.supportsGenerics = supportsGenerics;
  }
}
