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
package org.sonar.plugins.python.api.quickfix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.api.Beta;

@Beta
public class PythonQuickFix {
  private final String description;
  private final List<PythonTextEdit> textEdits;

  private PythonQuickFix(String description, List<PythonTextEdit> textEdits) {
    this.description = description;
    this.textEdits = textEdits;
  }

  public String getDescription() {
    return description;
  }

  public List<PythonTextEdit> getTextEdits() {
    return textEdits;
  }

  public static Builder newQuickFix(String description) {
    return new Builder(description);
  }

  public static PythonQuickFix newQuickFix(String description, PythonTextEdit... textEdits) {
    return newQuickFix(description).addTextEdit(textEdits).build();
  }

  public static class Builder {
    private final String description;
    private final List<PythonTextEdit> textEdits = new ArrayList<>();

    private Builder(String description) {
      this.description = description;
    }

    public Builder addTextEdit(PythonTextEdit... textEdits) {
      return addTextEdit(Arrays.asList(textEdits));
    }

    public Builder addTextEdit(List<PythonTextEdit> textEdits) {
      this.textEdits.addAll(textEdits);
      return this;
    }

    public PythonQuickFix build() {
      return new PythonQuickFix(description, textEdits);
    }
  }

}
