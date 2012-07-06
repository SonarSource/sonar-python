package org.sonar.plugins.python.colorizer;

import org.junit.Test;
import org.sonar.colorizer.CodeColorizer;

import java.io.StringReader;

import static org.fest.assertions.Assertions.assertThat;

public class PythonColorizerTest {

  private PythonColorizer pythonColorizer = new PythonColorizer();
  private CodeColorizer codeColorizer = new CodeColorizer(pythonColorizer.getTokenizers());

  private String colorize(String sourceCode) {
    return codeColorizer.toHtml(new StringReader(sourceCode));
  }

  @Test
  public void increase_coverage_for_fun() {
    assertThat(pythonColorizer.getTokenizers()).isSameAs(pythonColorizer.getTokenizers());
  }

  @Test
  public void should_colorize_keywords() {
    assertThat(colorize("False")).contains("<span class=\"k\">False</span>");
  }

  @Test
  public void should_colorize_comments() {
    assertThat(colorize("# comment \n new line")).contains("<span class=\"cd\"># comment </span>");
  }

  @Test
  public void should_colorize_shortstring_literals() {
    assertThat(colorize("\"string\"")).contains("<span class=\"s\">\"string\"</span>");
  }

  @Test
  public void should_colorize_longstring_literals() {
    assertThat(colorize("\"\"\"string\"\"\"")).contains("<span class=\"s\">\"\"\"string\"\"\"</span>");
  }

}
