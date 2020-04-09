/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python.api;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;

import static java.lang.Double.MAX_VALUE;
import static java.lang.Double.MIN_VALUE;
import static java.lang.Double.max;
import static java.lang.Double.min;

public class PythonVersion {

  private static final double MIN_VERSION = 2.5;
  private static final double MAX_VERSION = 3.8;
  private static final Pattern PATTERN = Pattern.compile("^\\s*([<>]=?)\\s*([0-9]+(\\.[0-9]+)?)(\\.[0-9]+)?$");
  private static final Logger LOG = Loggers.get(PythonVersion.class);
  public static final String PYTHON_VERSION_KEY = "sonar.python.version";

  private double minVersion = MIN_VERSION;
  private double maxVersion = MAX_VERSION;

  private PythonVersion() {
  }

  public static PythonVersion fromString(String propertyValue) {
    String[] intervals = propertyValue.split(",");
    if (intervals.length > 2) {
      LOG.warn(String.format("Error while parsing value of parameter '%s' (%s). Only two intervals are supported (e.g. >= 3.6, < 3.8).", PYTHON_VERSION_KEY, propertyValue));
      return PythonVersion.allVersions();
    }
    PythonVersion pythonVersion = new PythonVersion();
    pythonVersion.minVersion = MIN_VALUE;
    pythonVersion.maxVersion = MAX_VALUE;
    for (String interval : intervals) {
      Matcher matcher = PATTERN.matcher(interval);
      if (!matcher.find()) {
        LOG.warn(String.format(Locale.ROOT, parseErrorMessage(propertyValue.trim()), PYTHON_VERSION_KEY, propertyValue, MIN_VERSION, MAX_VERSION));
        return PythonVersion.allVersions();
      }
      String operator = matcher.group(1);
      pythonVersion.setMinAndMaxVersion(operator, Double.parseDouble(matcher.group(2)));
    }
    if (pythonVersion.minVersion != MIN_VALUE && (pythonVersion.minVersion < MIN_VERSION || pythonVersion.minVersion > MAX_VERSION)) {
      LOG.warn(String.format(Locale.ROOT, "Python version range '%s' is not supported. Versions must be between %.1f and %.1f.", propertyValue, MIN_VERSION, MAX_VERSION));
      return PythonVersion.allVersions();
    }
    if (pythonVersion.maxVersion != MAX_VALUE && (pythonVersion.maxVersion > MAX_VERSION || pythonVersion.maxVersion < MIN_VERSION)) {
      LOG.warn(String.format(Locale.ROOT, "Python version range '%s' is not supported. Versions must be between %.1f and %.1f.", propertyValue, MIN_VERSION, MAX_VERSION));
      return PythonVersion.allVersions();
    }
    if (pythonVersion.minVersion > pythonVersion.maxVersion) {
      return PythonVersion.allVersions();
    }
    pythonVersion.minVersion = max(pythonVersion.minVersion, MIN_VERSION);
    pythonVersion.maxVersion = min(pythonVersion.maxVersion, MAX_VERSION);
    return pythonVersion;
  }

  protected static String parseErrorMessage(String propertyValue) {
    String prefix = "Error while parsing value of parameter '%s' (%s). ";
    if (propertyValue.startsWith(">") || propertyValue.startsWith("<")) {
      return prefix + "Versions must be between %.1f and %.1f.";
    }
    return prefix + "Only intervals are supported (e.g. >= 3.6, < 3.8).";
  }

  public static PythonVersion allVersions() {
    return new PythonVersion();
  }

  private void setMinAndMaxVersion(String operator, double version) {
    if (operator.equals(">=")) {
      minVersion = max(minVersion, version);
      maxVersion = min(maxVersion, MAX_VALUE);
    }
    if (operator.equals(">")) {
      minVersion = max(minVersion, version + 0.1);
      maxVersion = min(maxVersion, MAX_VALUE);
    }
    if (operator.equals("<=")) {
      minVersion = max(minVersion, MIN_VALUE);
      maxVersion = min(maxVersion, version);
    }
    if (operator.equals("<")) {
      minVersion = max(minVersion, MIN_VALUE);
      maxVersion = min(maxVersion, version - 0.1);
    }
  }

  public boolean isPython2Only() {
    return maxVersion < 3;
  }

  public boolean isPython3Only() {
    return minVersion >= 3;
  }
}
