import org.antlr.runtime.*;

public class TestParser {

  // override nextToken to set startPos
  public static class MyLexer extends PythonLexer {

    public MyLexer(CharStream lexer) {
      super(lexer);
    }

    public Token nextToken() {
      startPos = getCharPositionInLine();
      return super.nextToken();
    }
  }

  public static void main(String[] args) throws Exception {
    PythonLexer lexer = new MyLexer(new ANTLRFileStream(args[0]));
    CommonTokenStream tokens = new CommonTokenStream(lexer);
    PythonTokenSource indentedSource = new PythonTokenSource(tokens);
    tokens = new CommonTokenStream(indentedSource);
    PythonParser parser = new PythonParser(tokens);
    parser.file_input();
  }
}
