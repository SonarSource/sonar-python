from flask_sqlalchemy import SQLAlchemy

def foo():
    db = SQLAlchemy(app)
    class Example1(db.Model):
        __tablename__ = 'example1'
        uuid = db.Column(db.Integer, primary_key=True)
        # `backref` will necessarily create a circular reference: it create an implicite relationship named 'parent' in ChildBackref
        child = db.relationship('Child', backref='parent') # This relationship creates circular references
