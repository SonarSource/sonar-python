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

import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.LinkedList;
import java.util.List;
import java.io.*;
import org.sonar.api.utils.SonarException;

public final class Utils {
  private static final Logger LOGGER = LoggerFactory.getLogger(Utils.class);
  
  public static List<String> callCommand(String command) {
    List<String> lines = new LinkedList<String>();
    
    LOGGER.debug("Calling command: '{}'", command);
    
    InputStream is = null;
    try {
      Process p = Runtime.getRuntime().exec(command);
      is = p.getInputStream();
      BufferedReader stdInput = new BufferedReader(new InputStreamReader(is));
      String s = null;
      
      while ((s = stdInput.readLine()) != null) {
	lines.add(s);
      }
    } catch (IOException e) {
      throw new SonarException("Error calling command", e);
    } finally {
      IOUtils.closeQuietly(is);
    }
    
    return lines;
  }
  
  private Utils() {}
}
