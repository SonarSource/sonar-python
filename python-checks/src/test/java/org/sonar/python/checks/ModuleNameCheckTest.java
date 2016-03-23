/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.python.checks;

import org.junit.Rule;
import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;
import org.sonar.squidbridge.checks.CheckMessagesVerifierRule;

import java.io.File;

public class ModuleNameCheckTest {

  private ModuleNameCheck check = new ModuleNameCheck();
  @Rule
  public CheckMessagesVerifierRule checkMessagesVerifier = new CheckMessagesVerifierRule();

  @Test
  public void bad_name() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/badModule_name.py"), check);
    String format = "(([a-z_][a-z0-9_]*)|([A-Z][a-zA-Z0-9]+))$";
    String message = "Rename this module to match this regular expression: \"%s\".";
    checkMessagesVerifier.verify(file.getCheckMessages())
        .next().withMessage(String.format(message, format));
  }

  @Test
  public void good_name_camel_case() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/ModuleName.py"), check);
    checkMessagesVerifier.verify(file.getCheckMessages());
  }

  @Test
  public void good_name_snake_case() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/module_name.py"), check);
    checkMessagesVerifier.verify(file.getCheckMessages());
  }

}
