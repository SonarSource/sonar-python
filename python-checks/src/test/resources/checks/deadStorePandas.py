import pandas as pd

def foo():
    area_of_interest = 'fictional_city' # OK
    pop = sample_df.query('area == @area_of_interest').population
    area_of_interest = 'other'
    something(area_of_interest)

def bar():
    abc = 'fictional_city' # Noncompliant
    pop = sample_df.query('area == @area_of_interest').population
    abc = 'other'
    something(abc)
