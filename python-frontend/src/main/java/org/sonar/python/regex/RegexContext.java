package org.sonar.python.regex;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;

public interface RegexContext {

  RegexParseResult regexForStringElement(StringElement stringElement);

  PythonCheck.PreciseIssue addIssue(Tree element, @Nullable String message);
}
