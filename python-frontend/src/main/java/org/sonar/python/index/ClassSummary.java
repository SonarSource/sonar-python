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
import org.sonar.plugins.python.api.LocationInFile;

public class ClassSummary implements Summary {

  private final String name;
  private final String fullyQualifiedName;
  private final Collection<String> superClasses;
  private final Collection<Summary> members;
  private final boolean hasDecorators;
  private final LocationInFile definitionLocation;
  private final boolean hasSuperClassWithoutSymbol;
  private final boolean hasMetaClass;
  private final String metaclassFQN;
  private final boolean supportsGenerics;

  public ClassSummary(String name, String fullyQualifiedName, Collection<String> superClasses, Collection<Summary> members,
    boolean hasDecorators, LocationInFile definitionLocation, boolean hasSuperClassWithoutSymbol, boolean hasMetaClass, String metaclassFQN, boolean supportsGenerics) {

    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.superClasses = superClasses;
    this.members = members;
    this.hasDecorators = hasDecorators;
    this.definitionLocation = definitionLocation;
    this.hasSuperClassWithoutSymbol = hasSuperClassWithoutSymbol;
    this.hasMetaClass = hasMetaClass;
    this.metaclassFQN = metaclassFQN;
    this.supportsGenerics = supportsGenerics;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public Collection<String> superClasses() {
    return superClasses;
  }

  public Collection<Summary> members() {
    return members;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  public boolean hasSuperClassWithoutSymbol() {
    return hasSuperClassWithoutSymbol;
  }

  public LocationInFile definitionLocation() {
    return definitionLocation;
  }

  public boolean hasMetaClass() {
    return false;
  }

  public String metaclassFQN() {
    return null;
  }

  public boolean supportsGenerics() {
    return false;
  }

  public static class ClassSummaryBuilder {

    private String name;
    private String fullyQualifiedName;
    private Collection<String> superClasses;
    private Collection<Summary> members;
    private boolean hasDecorators;
    private LocationInFile definitionLocation;
    private boolean hasSuperClassWithoutSymbol;
    private boolean hasMetaClass;
    private String metaclassFQN;
    private boolean supportsGenerics;

    public ClassSummaryBuilder withName(String name) {
      this.name = name;
      return this;
    }

    public ClassSummaryBuilder withFullyQualifiedName(String fullyQualifiedName) {
      this.fullyQualifiedName = fullyQualifiedName;
      return this;
    }

    public ClassSummaryBuilder withSuperClasses(Collection<String> superClasses) {
      this.superClasses = superClasses;
      return this;
    }

    public ClassSummaryBuilder withMembers(Collection<Summary> members) {
      this.members = members;
      return this;
    }

    public ClassSummaryBuilder withHasDecorators(boolean hasDecorators) {
      this.hasDecorators = hasDecorators;
      return this;
    }

    public ClassSummaryBuilder withHasSuperClassWithoutSymbol(boolean hasSuperClassWithoutSymbol) {
      this.hasSuperClassWithoutSymbol = hasSuperClassWithoutSymbol;
      return this;
    }

    public ClassSummaryBuilder withDefinitionLocation(LocationInFile definitionLocation) {
      this.definitionLocation = definitionLocation;
      return this;
    }

    public ClassSummaryBuilder withHasMetaClass(boolean hasMetaClass) {
      this.hasMetaClass = hasMetaClass;
      return this;
    }

    public ClassSummaryBuilder withMetaclassFQN(String metaclassFQN) {
      this.metaclassFQN = metaclassFQN;
      return this;
    }

    public ClassSummaryBuilder withSupportsGenerics(boolean supportsGenerics) {
      this.supportsGenerics = supportsGenerics;
      return this;
    }

    public ClassSummary build() {
      return new ClassSummary(name, fullyQualifiedName, superClasses, members, hasDecorators, definitionLocation, hasSuperClassWithoutSymbol, hasMetaClass, metaclassFQN, supportsGenerics);
    }
  }
}
