/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */

package org.sonar.plugins.python;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.sonar.api.utils.SonarException;

public final class Utils {
  public static List<String> callCommand(String command, String[] environ) {
    List<String> lines = new LinkedList<String>();

    PythonPlugin.LOG.debug("Calling command: '{}'", command);

    BufferedReader stdInput = null;
    try {
      Process p = null;
      if(environ == null){
        p = Runtime.getRuntime().exec(command);
      } else {
        p = Runtime.getRuntime().exec(command, environ);
      }

      stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
      String s = null;

      while ((s = stdInput.readLine()) != null) {
        lines.add(s);
      }
    } catch (IOException e) {
      throw new SonarException("Error calling command '" + command +
                               "', details: '" + e + "'");
    } finally {
      IOUtils.closeQuietly(stdInput);
    }

    return lines;
  }

  private Utils() {
  }
}
