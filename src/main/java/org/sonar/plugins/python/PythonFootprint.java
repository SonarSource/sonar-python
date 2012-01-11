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

import java.util.HashSet;
import java.util.Set;

import org.sonar.squid.recognizer.Detector;
import org.sonar.squid.recognizer.KeywordsDetector;
import org.sonar.squid.recognizer.LanguageFootprint;

public class PythonFootprint implements LanguageFootprint {

  public Set<Detector> getDetectors() {
    // Doesn't play a significant role, apparently...
    // So live for now with this simple implementation
    //

    final Set<Detector> detectors = new HashSet<Detector>();

    // wild guess...
    // detectors.add(new EndWithDetector(0.3, ':', ')'));

    detectors.add(new KeywordsDetector(0.3, Python.KEYWORDS));

    return detectors;
  }
}
