/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Token;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.Nullable;

public abstract class PythonCheckAstNode extends PythonVisitor implements PythonCheck {

  protected final PreciseIssue addIssue(AstNode node, @Nullable String message) {
    PreciseIssue newIssue = new PreciseIssue(this, IssueLocation.preciseLocation(node, message));
    getContext().addIssue(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(IssueLocation primaryLocation) {
    PreciseIssue newIssue = new PreciseIssue(this, primaryLocation);
    getContext().addIssue(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addLineIssue(String message, int lineNumber) {
    PreciseIssue newIssue = new PreciseIssue(this, IssueLocation.atLineLevel(message, lineNumber));
    getContext().addIssue(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addFileIssue(String message) {
    PreciseIssue newIssue = new PreciseIssue(this, IssueLocation.atFileLevel(message));
    getContext().addIssue(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(Token token, String message) {
    return addIssue(new AstNode(token), message);
  }

  public static <T> Set<T> immutableSet(T... el) {
    return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(el)));
  }
}
