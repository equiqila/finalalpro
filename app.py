import os
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model dan nama fitur
model = joblib.load('xgb_model.pkl')
feature_names = joblib.load('feature_names.pkl')  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['csv_file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
            file.save(filepath)

            # Baca file asli
            data = pd.read_csv(filepath, sep=";")

            # Validasi kolom yang dibutuhkan
            required_columns = [
                'customer_id', 'nama', 'age', 'location', 'subscription_type', 'payment_plan',
                'num_subscription_pauses', 'payment_method', 'customer_service_inquiries',
                'signup_date', 'weekly_hours', 'average_session_length', 'song_skip_rate',
                'weekly_songs_played', 'weekly_unique_songs', 'num_favorite_artists',
                'num_platform_friends', 'num_playlists_created', 'num_shared_playlists',
                'notifications_clicked'
            ]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return render_template('predict.html', message=f"Missing columns: {', '.join(missing_columns)}")

            # Simpan kolom customer_id dan nama
            nama_column = data['nama']
            customer_id = data['customer_id']

            # Pilih hanya kolom fitur
            training_features = [
                'customer_id', 'age', 'location', 'subscription_type', 'payment_plan', 'num_subscription_pauses',
                'payment_method', 'customer_service_inquiries', 'signup_date', 'weekly_hours',
                'average_session_length', 'song_skip_rate', 'weekly_songs_played',
                'weekly_unique_songs', 'num_favorite_artists', 'num_platform_friends',
                'num_playlists_created', 'num_shared_playlists', 'notifications_clicked'
            ]
            X = data[training_features]
            

            # Lakukan Label Encoding untuk kolom kategori
            label_encoder = LabelEncoder()
            for col in X.select_dtypes(include=['object']).columns:
                X[col] = label_encoder.fit_transform(X[col])

            # Prediksi
            predictions = model.predict(X)
            predictions_text = ['not potentially churned' if pred == 0 else 'potentially churned' for pred in predictions]

            # Gabungkan hasil prediksi ke data asli
            data['prediction'] = predictions_text
            data['prediction_value'] = predictions

            # Simpan file gabungan
            full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_predictions.csv')
            data.to_csv(full_filepath, index=False)

            # Tampilkan hasil singkat
            result_df = pd.DataFrame({'customer_id': customer_id, 'nama': nama_column, 'prediction': predictions_text})
            return render_template('predict.html', predictions=result_df.to_html(classes='table table-striped', index=False))
    return render_template('predict.html')

@app.route('/model_explainability')
def model_explainability():
    try:
        # Ambil feature importance dari model
        feature_importances = model.feature_importances_

        # Urutkan feature importance dari terbesar ke terkecil
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_features = np.array(feature_names)[sorted_indices]
        sorted_importances = feature_importances[sorted_indices]

        # Siapkan data untuk dikirim ke template
        importance_data = {
            "features": sorted_features.tolist(),
            "importances": sorted_importances.tolist()
        }

        return render_template('model_explainability.html', importance_data=importance_data)

    except Exception as e:
        return render_template(
            'model_explainability.html',
            importance_data={},
            error=f"Error processing feature importance: {str(e)}"
        )

@app.route('/dashboard')
def dashboard():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'latest_predictions.csv')
    if not os.path.exists(filepath):
        return render_template(
            'dashboard.html',
            churn_count=0,
            non_churn_count=0,
            subscription_churn={},
            churn_by_age={},
            churn_by_payment_method={},
            error="Please upload data first."
        )

    try:
        # Baca file gabungan
        data = pd.read_csv(filepath)

        # Bersihkan kolom numerik dari format string
        numeric_columns = ['weekly_hours', 'average_session_length', 'song_skip_rate',
                           'weekly_songs_played', 'weekly_unique_songs',
                           'num_favorite_artists', 'num_platform_friends',
                           'num_playlists_created', 'num_shared_playlists',
                           'notifications_clicked', 'age']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Hitung churn dan non-churn
        churn_count = data['prediction_value'].sum()
        non_churn_count = len(data) - churn_count

        # Churn by subscription_type
        subscription_churn = data.groupby('subscription_type')['prediction_value'].sum().to_dict()

        # Age grouping per 5 years
        data['age_group'] = pd.cut(data['age'], bins=range(0, 100, 5), right=False)
        churn_by_age = data.groupby(data['age_group'].astype(str))['prediction_value'].sum().to_dict()

        # Churn by payment method
        churn_by_payment_method = data.groupby('payment_method')['prediction_value'].sum().to_dict()

        return render_template(
            'dashboard.html',
            churn_count=churn_count,
            non_churn_count=non_churn_count,
            subscription_churn=subscription_churn,
            churn_by_age=churn_by_age,
            churn_by_payment_method=churn_by_payment_method
        )
    except Exception as e:
        return render_template(
            'dashboard.html',
            churn_count=0,
            non_churn_count=0,
            subscription_churn={},
            churn_by_age={},
            churn_by_payment_method={},
            error=f"Error processing data: {str(e)}"
        )


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Buat folder uploads jika belum ada
    app.run(debug=True)
