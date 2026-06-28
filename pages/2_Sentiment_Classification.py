import streamlit as st
from utils.theme import set_page_config
from utils.helper import load_dataset
from utils.feature_engineering import prepare_dataset, prepare_tfidf, prepare_train_test
from utils.analytics_classification import train_all_models, evaluate_models, get_classification_report, get_confusion_matrix, predict_text
from utils.plot_sentiment import plot_sentiment_pie, plot_sentiment_bar_brand, plot_timeline, plot_model_comparison, plot_confusion_matrix_heatmap
import pandas as pd

set_page_config(page_title="Sentiment & Classification", layout="wide")

def main():
    st.title("🧩 Sentiment Analysis & Model Classification")
    st.write("Menganalisis persepsi publik berbasis Lexicon InSet dan membandingkan performa model Machine Learning.")
    
    try:
        # Load & Prepare Data
        df_raw = load_dataset()
        df = prepare_dataset(df_raw)
        
        # --- SIDEBAR FILTER ---
        st.sidebar.header("Filter Analisis")
        brands = df['Brand'].unique().tolist()
        selected_brands = st.sidebar.multiselect("Pilih Brand", brands, default=brands)
        
        sentiments = df['inset_sentiment'].unique().tolist()
        selected_sentiments = st.sidebar.multiselect("Pilih Sentimen", sentiments, default=sentiments)
        
        # Apply filter
        df_filtered = df[(df['Brand'].isin(selected_brands)) & (df['inset_sentiment'].isin(selected_sentiments))]
        
        # --- TAB 1: EKSPLORASI SENTIMEN ---
        # --- TAB 2: KLASIFIKASI MACHINE LEARNING ---
        tab1, tab2 = st.tabs(["📊 Eksplorasi Sentimen", "🤖 Klasifikasi Machine Learning"])
        
        with tab1:
            st.subheader("Distribusi Sentimen (Lexicon InSet)")
            
            col1, col2 = st.columns([1, 1.5])
            with col1:
                if not df_filtered.empty:
                    st.plotly_chart(plot_sentiment_pie(df_filtered), use_container_width=True)
                else:
                    st.warning("Data kosong untuk filter yang dipilih.")
            with col2:
                if not df_filtered.empty:
                    st.plotly_chart(plot_sentiment_bar_brand(df_filtered), use_container_width=True)
            
            st.markdown("---")
            if 'created_at' in df_filtered.columns:
                st.plotly_chart(plot_timeline(df_filtered), use_container_width=True)
                
        with tab2:
            st.subheader("Evaluasi Model Prediktif")
            
            # --- MANAJEMEN SESSION STATE ---
            if "model_trained" not in st.session_state:
                st.session_state.model_trained = False
                st.session_state.models = None
                st.session_state.metrics = None
                st.session_state.X_test = None
                st.session_state.y_test = None
                st.session_state.tfidf_vectorizer = None

            with st.form(key="ml_training_form"):
                st.markdown("##### ⚙️ Konfigurasi Parameter Training")
                
                col_form1, col_form2 = st.columns(2)
                with col_form1:
                    max_features_input = st.number_input("Max Features TF-IDF", min_value=500, max_value=5000, value=1500)
                with col_form2:
                    test_size_input = st.slider("Test Size Ratio", min_value=0.1, max_value=0.5, value=0.2)
                
                submit_button = st.form_submit_button(label="🚀 Jalankan Komparasi Model")
            
                if submit_button:
                    with st.spinner("Mengekstraksi TF-IDF dan melatih model (Proses ini diisolasi)..."):
                        # Masukkan input UI ke dalam parameter fungsi!
                        X_tfidf, tfidf_vectorizer = prepare_tfidf(df['clean_text'], max_features=max_features_input) 
                        X_train, X_test, y_train, y_test = prepare_train_test(X_tfidf, df['inset_sentiment'], test_size=test_size_input)
                    
                    models = train_all_models(X_train, y_train)
                    metrics = evaluate_models(models, X_test, y_test)
                    
                    st.session_state.models = models
                    st.session_state.metrics = metrics
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.tfidf_vectorizer = tfidf_vectorizer
                    st.session_state.model_trained = True

            if st.session_state.model_trained:
                metrics = st.session_state.metrics
                models = st.session_state.models
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                tfidf_vectorizer = st.session_state.tfidf_vectorizer

                col_chart, col_rank = st.columns([2, 1])
                with col_chart:
                    st.plotly_chart(plot_model_comparison(metrics), use_container_width=True)
                with col_rank:
                    st.markdown("### 🏆 Ranking Model")
                    for i, (m_name, acc) in enumerate(sorted(metrics.items(), key=lambda x:x[1], reverse=True), 1):
                        st.success(f"**{i}. {m_name}** : {acc*100:.2f}%")
                
                st.markdown("---")
                
                selected_model_name = st.selectbox("Pilih Model untuk melihat Detail (Classification Report & Confusion Matrix):", list(models.keys()))
                selected_model = models[selected_model_name]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Classification Report ({selected_model_name})**")
                    report_df = get_classification_report(selected_model, selected_model_name, X_test, y_test)
                    st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                    
                with c2:
                    cm_df = get_confusion_matrix(selected_model, selected_model_name, X_test, y_test)
                    st.plotly_chart(plot_confusion_matrix_heatmap(cm_df, selected_model_name), use_container_width=True)
                    
                st.markdown("---")
                
                # Fitur Prediksi Mandiri dengan perlindungan "Unknown"
                st.subheader("🧪 Uji Prediksi Teks")
                user_input = st.text_input("Masukkan teks review/komentar baru:")
                if st.button("Prediksi Sentimen"):
                    if user_input.strip() == "":
                        st.warning("Masukkan teks terlebih dahulu.")
                    else:
                        prediction = predict_text(user_input, selected_model, selected_model_name, tfidf_vectorizer)
                        
                        if prediction == 'Positive':
                            st.success(f"Hasil Prediksi ({selected_model_name}): **Positif** 🟢")
                        elif prediction == 'Negative':
                            st.error(f"Hasil Prediksi ({selected_model_name}): **Negatif** 🔴")
                        elif prediction == 'Unknown':
                            st.warning("⚠️ **Tidak Dikenali:** Kata-kata yang Anda masukkan tidak ada dalam dataset pelatihan model (Out of Vocabulary). Silakan gunakan kalimat lain.")
                        else:
                            st.info(f"Hasil Prediksi ({selected_model_name}): **Netral** ⚪")
            else:
                st.info("💡 Silakan klik tombol **'🚀 Jalankan Komparasi Model'** di atas untuk memulai kalkulasi Machine Learning.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")

if __name__ == "__main__":
    main()