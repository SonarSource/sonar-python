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
package org.sonar.python.index;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.TypeShed;

public class SymbolBuilder {
  private Collection<Descriptor> descriptors = Collections.emptySet();
  private final Map<String, Symbol> existingSymbols;
  private final ProjectDescriptor projectDescriptor;
  @Nullable
  private String fullyQualifiedName;
  private String alias;

  public SymbolBuilder(Map<String, Symbol> existingSymbols, ProjectDescriptor projectDescriptor) {
    this.existingSymbols = existingSymbols;
    this.projectDescriptor = projectDescriptor;
  }

  public SymbolBuilder fromDescriptors(Collection<Descriptor> descriptors) {
    this.descriptors = descriptors;
    return this;
  }

  public SymbolBuilder fromFullyQualifiedName(@Nullable String fullyQualifiedName) {
    this.fullyQualifiedName = fullyQualifiedName;
    for (ModuleDescriptor module : projectDescriptor.modules().values()) {
      descriptors = module.descriptorsWithFQN(fullyQualifiedName);
      if (!descriptors.isEmpty()) {
        return this;
      }
    }
    return this;
  }

  public SymbolBuilder havingAlias(String alias) {
    this.alias = alias;
    return this;
  }

  @CheckForNull
  public Symbol build() {
    Set<Symbol> symbols = new HashSet<>();
    for (Descriptor descriptor : descriptors) {
      String name = alias != null ? alias : descriptor.name();
      if (descriptor instanceof FunctionDescriptor) {
        symbols.add(new FunctionSymbolImpl(name, ((FunctionDescriptor) descriptor)));
      } else if (descriptor.kind().equals(Descriptor.Kind.CLASS)) {
        symbols.add(classSymbol(name, ((ClassDescriptor) descriptor)));
      } else if (descriptor instanceof VariableDescriptor) {
        symbols.add(new SymbolImpl(name, descriptor.fullyQualifiedName(), ((VariableDescriptor) descriptor).annotatedType()));
      }
    }
    if (symbols.isEmpty()) {
      return fullyQualifiedName != null ? TypeShed.symbolWithFQN(fullyQualifiedName) : null;
    }
    if (symbols.size() == 1) {
      return symbols.iterator().next();
    }
    return AmbiguousSymbolImpl.create(symbols);
  }

  private Symbol classSymbol(String name, ClassDescriptor classDescriptor) {
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(name, classDescriptor.fullyQualifiedName(), classDescriptor.definitionLocation(),
      classDescriptor.hasDecorators(), classDescriptor.hasMetaClass(), classDescriptor.metaclassFQN(), classDescriptor.supportsGenerics());
    if (classDescriptor.hasSuperClassWithoutDescriptor()) {
      classSymbol.setHasSuperClassWithoutSymbol();
    }
    addSuperClasses(classDescriptor, classSymbol);
    addMembers(classDescriptor, classSymbol);
    return classSymbol;
  }

  private void addSuperClasses(ClassDescriptor classDescriptor, ClassSymbolImpl classSymbol) {
    for (String superClass : classDescriptor.superClasses()) {
      Symbol superClassSymbol = existingSymbols.get(superClass);
      if (superClassSymbol == null) {
        superClassSymbol = new SymbolBuilder(existingSymbols, projectDescriptor)
          .fromFullyQualifiedName(superClass)
          .build();
      }
      if (superClassSymbol != null) {
        classSymbol.addSuperClass(superClassSymbol);
      } else {
        classSymbol.setHasSuperClassWithoutSymbol();
      }
    }
  }

  private void addMembers(ClassDescriptor classDescriptor, ClassSymbolImpl classSymbol) {
    Set<Symbol> members = new HashSet<>();
    for (Descriptor member : classDescriptor.members()) {
      Symbol memberSymbol = new SymbolBuilder(existingSymbols, projectDescriptor)
        .fromDescriptors(Collections.singleton(member))
        .build();
      members.add(memberSymbol);
    }
    classSymbol.addMembers(members);
  }
}
