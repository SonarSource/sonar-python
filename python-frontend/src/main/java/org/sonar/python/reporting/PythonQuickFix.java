/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.reporting;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.IssueLocation;

public class PythonQuickFix {
  private final String description;
  private final List<IssueLocation.PythonTextEdit> textEdits;

  private PythonQuickFix(String description, List<IssueLocation.PythonTextEdit> textEdits) {
    this.description = description;
    this.textEdits = textEdits;
  }

  public String getDescription() {
    return description;
  }

  public List<IssueLocation.PythonTextEdit> getTextEdits() {
    return textEdits;
  }

  public static Builder newQuickFix(String description) {
    return new Builder(description);
  }

  public static class Builder {
    private final String description;
    private final List<IssueLocation.PythonTextEdit> textEdits = new ArrayList<>();

    private Builder(String description) {
      this.description = description;
    }

    public Builder addTextEdit(IssueLocation.PythonTextEdit... textEdit) {
      textEdits.addAll(Arrays.asList(textEdit));
      return this;
    }

    public Builder addTextEdits(List<IssueLocation.PythonTextEdit> textEdits) {
      this.textEdits.addAll(textEdits);
      return this;
    }

    public PythonQuickFix build() {
      return new PythonQuickFix(description, textEdits);
    }
  }

}
