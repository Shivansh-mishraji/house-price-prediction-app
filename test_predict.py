import pandas as pd
import pickle

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    input_data = pd.DataFrame({
        'sqft_living': [2000],
        'sqft_lot': [7500],
        'bedrooms': [3],
        'bathrooms': [2.0],
        'floors': [1],
        'view': [0],
        'condition': [3],
        'grade': [7],
        'sqft_above': [1600],
        'sqft_basement': [400],
        'yr_built': [1995],
        'yr_renovated': [0],
        'garage': [1],
        'parking': [2],
        'hoa_monthly': [0],
        'lat': [19.0760],
        'long': [72.8777],
        'zipcode': [400001],
        'city': ['Mumbai'],
        'neighborhood': ['Central'],
        'waterfront': [0]
    })

    prediction = model.predict(input_data)[0]
    print("Success. Prediction:", prediction)
except Exception as e:
    import traceback
    traceback.print_exc()
