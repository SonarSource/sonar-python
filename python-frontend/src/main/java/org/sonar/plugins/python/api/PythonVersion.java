/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;

import static org.sonar.plugins.python.api.PythonVersion.Version.V_27;
import static org.sonar.plugins.python.api.PythonVersion.Version.V_35;
import static org.sonar.plugins.python.api.PythonVersion.Version.V_36;
import static org.sonar.plugins.python.api.PythonVersion.Version.V_37;
import static org.sonar.plugins.python.api.PythonVersion.Version.V_38;
import static org.sonar.plugins.python.api.PythonVersion.Version.V_39;

public class PythonVersion {

  public enum Version {
    V_27(2.7), V_35(3.5), V_36(3.6), V_37(3.7), V_38(3.8), V_39(3.9);

    private final double value;
    Version(double value) {
      this.value = value;
    }

    public double value() {
      return value;
    }
  }

  private static final Map<String, Version> STRING_VERSION_MAP = new HashMap<>();
  static {
    STRING_VERSION_MAP.put("2", V_27);
    STRING_VERSION_MAP.put("2.7", V_27);
    STRING_VERSION_MAP.put("3", V_35);
    STRING_VERSION_MAP.put("3.0", V_35);
    STRING_VERSION_MAP.put("3.1", V_35);
    STRING_VERSION_MAP.put("3.2", V_35);
    STRING_VERSION_MAP.put("3.3", V_35);
    STRING_VERSION_MAP.put("3.4", V_35);
    STRING_VERSION_MAP.put("3.5", V_35);
    STRING_VERSION_MAP.put("3.6", V_36);
    STRING_VERSION_MAP.put("3.7", V_37);
    STRING_VERSION_MAP.put("3.8", V_38);
    STRING_VERSION_MAP.put("3.9", V_39);
  }
  private static final Version MIN_SUPPORTED_VERSION = V_27;
  private static final Version MAX_SUPPORTED_VERSION = V_39;
  private static final Logger LOG = Loggers.get(PythonVersion.class);
  public static final String PYTHON_VERSION_KEY = "sonar.python.version";

  private List<Version> supportedVersions = new ArrayList<>();

  private PythonVersion() {
  }

  public static PythonVersion fromString(String propertyValue) {
    String[] versions = propertyValue.split(",");
    if (versions.length == 0) {
      return allVersions();
    }
    PythonVersion pythonVersion = new PythonVersion();
    for (String versionValue : versions) {
      versionValue = versionValue.trim();
      Version version = STRING_VERSION_MAP.get(versionValue);
      if (version != null) {
        pythonVersion.addVersion(version);
      } else {
        boolean isGuessSuccessful = guessPythonVersion(pythonVersion, versionValue);
        if (!isGuessSuccessful) {
          return allVersions();
        }
      }
    }
    return pythonVersion;
  }

  public static PythonVersion allVersions() {
    PythonVersion pythonVersion = new PythonVersion();
    pythonVersion.supportedVersions = Arrays.asList(V_27, V_35, V_36, V_37, V_38, V_39);
    return pythonVersion;
  }

  private static boolean guessPythonVersion(PythonVersion pythonVersion, String versionValue) {
    try {
      double parsedVersion = Double.parseDouble(versionValue);
      if (parsedVersion < MIN_SUPPORTED_VERSION.value()) {
        pythonVersion.addVersion(MIN_SUPPORTED_VERSION);
      } else if (parsedVersion > MAX_SUPPORTED_VERSION.value()) {
        pythonVersion.addVersion(MAX_SUPPORTED_VERSION);
      } else {
        logErrorMessage(versionValue);
        return false;
      }
    } catch (NumberFormatException nfe) {
      logErrorMessage(versionValue);
      return false;
    }
    return true;
  }

  private static void logErrorMessage(String propertyValue) {
    String prefix = "Error while parsing value of parameter '%s' (%s). Versions must be specified as MAJOR_VERSION.MIN.VERSION (e.g. \"3.7, 3.8\")";
    LOG.warn(String.format(Locale.ROOT, prefix, PYTHON_VERSION_KEY, propertyValue));
  }

  private void addVersion(Version version) {
    supportedVersions.add(version);
  }

  public List<Version> supportedVersions() {
    return supportedVersions;
  }
}
