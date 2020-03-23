def get_numberic_columns(data_frame):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return data_frame.select_dtypes(include=numerics)
    
    
def fill_na(data_frame, value = 0):
    for column in data_frame.columns:
        data_frame[column].fillna(value)
    return data_frame
