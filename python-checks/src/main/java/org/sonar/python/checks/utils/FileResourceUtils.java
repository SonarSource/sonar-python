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
package org.sonar.python.checks.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.Set;

public class FileResourceUtils {
  private FileResourceUtils(){}

  public static Set<String> loadResourceAsSet(String resourceName, Charset charset) throws IOException {
    try (InputStream is = FileResourceUtils.class.getResourceAsStream(resourceName)) {
      if (is == null) {
        throw new MissingResourceException("Cannot find resource file '" + resourceName + "'");
      }
      try (InputStreamReader isr = new InputStreamReader(is, charset);
           BufferedReader br = new BufferedReader(isr)) {
        Set<String> result = new HashSet<>();
        String line;
        while((line = br.readLine()) != null) {
          result.add(line);
        }
        return result;
      }
    }
  }

  public static class MissingResourceException extends RuntimeException {
    public MissingResourceException(Exception exception) {
      super(exception);
    }
    public MissingResourceException(String message) {
      super(message);
    }
  }
}
