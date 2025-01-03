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
package org.sonar.python.tree;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ImportFromImpl extends SimpleStatement implements ImportFrom {
  private final Token fromKeyword;
  private final List<Token> dottedPrefixForModule;
  private final DottedName moduleName;
  private final Token importKeyword;
  private final List<AliasedName> aliasedImportNames;
  private final boolean isWildcardImport;
  private final Token wildcard;
  private final Separators separators;
  private boolean hasUnresolvedWildcardImport = false;

  public ImportFromImpl(Token fromKeyword, @Nullable List<Token> dottedPrefixForModule, @Nullable DottedName moduleName,
                        Token importKeyword, @Nullable List<AliasedName> aliasedImportNames, @Nullable Token wildcard, Separators separators) {
    this.fromKeyword = fromKeyword;
    this.dottedPrefixForModule = dottedPrefixForModule;
    this.moduleName = moduleName;
    this.importKeyword = importKeyword;
    this.aliasedImportNames = aliasedImportNames == null ? Collections.emptyList() : aliasedImportNames;
    this.wildcard = wildcard;
    this.isWildcardImport = wildcard != null;
    this.separators = separators;
  }

  @Override
  public Token fromKeyword() {
    return fromKeyword;
  }

  @CheckForNull
  @Override
  public DottedName module() {
    return moduleName;
  }

  @Override
  public Token importKeyword() {
    return importKeyword;
  }

  @CheckForNull
  @Override
  public List<Token> dottedPrefixForModule() {
    return dottedPrefixForModule;
  }

  @Override
  public List<AliasedName> importedNames() {
    return aliasedImportNames;
  }

  @Override
  public boolean isWildcardImport() {
    return isWildcardImport;
  }

  @CheckForNull
  @Override
  public Token wildcard() {
    return wildcard;
  }

  @Override
  public boolean hasUnresolvedWildcardImport() {
    return hasUnresolvedWildcardImport;
  }

  public void setHasUnresolvedWildcardImport(boolean hasUnresolvedWildcardImport) {
    this.hasUnresolvedWildcardImport = hasUnresolvedWildcardImport;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.IMPORT_FROM;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitImportFrom(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(fromKeyword), dottedPrefixForModule, Arrays.asList(moduleName, importKeyword), aliasedImportNames,
      Collections.singletonList(wildcard), separators.elements()).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
