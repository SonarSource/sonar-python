# mod.py
def get_exported(): 
  def f(): pass
  return f
__all__ = get_exported()
