import pandas as pd


def loadDataset(path,columns=['anno','keyword1','keyword2','relationship']):
    dt =pd.read_csv(path,delimiter='\t',encoding='utf-8')
    oldColumns = dt.columns
    dt = pd.concat([dt, oldColumns.to_frame().T])
    dt.columns = columns
    return dt
