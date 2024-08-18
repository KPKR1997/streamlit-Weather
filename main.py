import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
data = pd.read_csv('seattle-weather.csv')

# Prepare the data
X = data.drop(columns=['date', 'weather'])
y = data['weather']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)



# Streamlit app
st.title('Weather Prediction')

st.write('This app predicts weather conditions based on input features like precipitation, max/min temperature, and wind. (Data source: kaggle')

# Inputs
precipitation = st.slider('Precipitation (mm)', min_value=0.0, max_value=100.0, value=0.0, step=0.1)
temp_max = st.slider('Maximum Temperature (°C)', min_value=-50.0, max_value=50.0, value=10.0, step=0.1)
temp_min = st.slider('Minimum Temperature (°C)', min_value=-50.0, max_value=50.0, value=5.0, step=0.1)
wind = st.slider('Wind Speed (m/s)', min_value=0.0, max_value=50.0, value=5.0, step=0.1)


# Prediction
if st.button('Predict Weather'):
    input_data = pd.DataFrame({
        'precipitation': [precipitation],
        'temp_max': [temp_max],
        'temp_min': [temp_min],
        'wind': [wind]
    })

    prediction = model.predict(input_data)
    st.write(f'The predicted weather is " {prediction[0]} "')

# Model Accuracy
st.write(f'Model Accuracy is {accuracy_score(y_test, model.predict(X_test)):.2f}')
