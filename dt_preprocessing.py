import pandas as pd


def loadDatasetOne():
    dt =pd.read_csv('./Dataset/ds-1.tsv',delimiter='\t',encoding='utf-8')
    oldColumns = dt.columns
    dt = pd.concat([dt, oldColumns.to_frame().T])
    dt.columns = ['anno','keyword1','keyword2','relationship']
    return dt
