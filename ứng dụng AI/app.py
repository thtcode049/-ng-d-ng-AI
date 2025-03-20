from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)


# Load the trained model and bins
best_model = joblib.load('best_model.pkl')
bins_q = joblib.load('bins_q.pkl')
bins_p = joblib.load('bins_p.pkl')
bins_d = joblib.load('bins_d.pkl')
# Load X_train columns
X_train_columns = joblib.load('X_train_columns.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from the form
            QuantityInv = float(request.form['QuantityInv'])
            UnitPrice = float(request.form['UnitPrice'])
            Month = int(request.form['Month'])

            # Create a DataFrame with the input data
            new_data = pd.DataFrame({
                'QuantityInv': [QuantityInv],
                'UnitPrice': [UnitPrice],
                'Month': [Month]
            })

            # ... (rest of the feature engineering and prediction code is the same)

            # Tạo các đặc trưng còn thiếu trong new_data
            new_data['QuantityRange'] = pd.cut(new_data['QuantityInv'], bins=bins_q)
            new_data['PriceRange'] = pd.cut(new_data['UnitPrice'], bins=bins_p)
            new_data['DateRange'] = pd.cut(new_data['Month'], bins=bins_d, labels=['q1', 'q2', 'q3', 'q4'])

            # Chuyển đổi các đặc trưng phân loại thành biến giả
            new_data = pd.get_dummies(new_data, columns=['QuantityRange'], prefix='qr')
            new_data = pd.get_dummies(new_data, columns=['PriceRange'], prefix='pr')
            new_data = pd.get_dummies(new_data, columns=['DateRange'], prefix='dr')

            # Loại bỏ các cột không cần thiết
            new_data = new_data[X_train_columns]  # Giữ lại các cột trùng khớp với X_train

            # Make the prediction
            prediction = best_model.predict(new_data)[0]

            # Render the template with the prediction
            return render_template('result.html', prediction=round(prediction, 2))

        except Exception as e:
            return f"Error: {str(e)}"

    # Render the input form if it's a GET request
    return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)