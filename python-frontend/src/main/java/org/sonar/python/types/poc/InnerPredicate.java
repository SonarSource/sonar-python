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
package org.sonar.python.types.poc;

import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.UnionType;

public interface InnerPredicate extends RawInnerPredicate {
  // This interface could be made into a class.
  // However, this would have the consequences that implementations are no longer able to be records.
  default TriBool rawApply(PythonType pythonType) {
    if (pythonType instanceof UnionType unionType) {
      for (PythonType candidate : unionType.candidates()) {
        var result = applyOn(candidate);
        if (result.equals(TriBool.UNKNOWN)) {
          return TriBool.UNKNOWN;
        }
        if (result.equals(TriBool.FALSE)) {
          return TriBool.FALSE;
        }
      }
      return TriBool.TRUE;
    } else {
      return applyOn(pythonType);
    }
  }

  TriBool applyOn(PythonType pythonType);
}
