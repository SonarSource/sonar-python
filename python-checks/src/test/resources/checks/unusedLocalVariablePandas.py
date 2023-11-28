import pandas as pd

def foo():
    d = {'area': ['fictional_town', 'fictional_city'], 'population': [100, 2000]}
    sample_df = pd.DataFrame(data=d)

    area_of_interest = 'fictional_city' # OK
    pop = sample_df.query('area == @area_of_interest').population # Noncompliant
