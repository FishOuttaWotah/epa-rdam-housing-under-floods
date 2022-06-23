import pandas as pd

def read_CBS_excel(filepath:str, index_name:str, convert_to_Int64:bool=False):
    # reads excel file from Rotterdam Onderwijs010 site and processes
    dataframe = pd.read_excel(filepath, header=1,skipfooter=7,
                              index_col=0, na_values=['-'])
    dataframe.index.name = index_name
    if convert_to_Int64:
        dataframe = dataframe.astype('Int64')
    return dataframe
