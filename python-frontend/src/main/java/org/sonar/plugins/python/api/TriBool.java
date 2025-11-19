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
package org.sonar.plugins.python.api;

import org.sonar.api.Beta;

@Beta
public enum TriBool {
  TRUE,
  FALSE,
  UNKNOWN;


  /**
   * This method performs a conservative AND of TriBool results
   * It is meant to be used when collapsing the results of testing a single predicate against candidates of a UnionType
   * Therefore, this variation will return UNKNOWN whenever one of the operands is UNKNOWN
   */
  public TriBool conservativeAnd(TriBool triBool) {
    if (this.equals(triBool)) {
      return this;
    }
    if (this.equals(UNKNOWN) || triBool.equals(UNKNOWN)) {
      return UNKNOWN;
    }
    return FALSE;
  }

  /**
   * This method performs a logical AND of TriBool results
   * It is meant to be used when performing the logical combination of ANDed predicates against a given type
   * Therefore, this variation behaves like a standard logical "AND" and will return FALSE whenever one of the operands is FALSE
   */
  public TriBool logicalAnd(TriBool triBool) {
    if (this.equals(triBool)) {
      return this;
    }
    if (this.equals(FALSE) || triBool.equals(FALSE)) {
      return FALSE;
    }
    return UNKNOWN;
  }

  public TriBool or(TriBool triBool) {
    if (this.equals(triBool)) {
      return this;
    }
    if (this.equals(TRUE) || triBool.equals(TRUE)) {
      return TRUE;
    }
    return UNKNOWN;
  }

  public boolean isTrue() {
    return this.equals(TRUE);
  }

  public boolean isFalse() {
    return this.equals(FALSE);
  }

  public boolean isUnknown() {
    return this.equals(UNKNOWN);
  }
}
