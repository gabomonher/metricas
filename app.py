import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar el modelo entrenado
# Asegúrate de que 'best_decision_tree_model.pkl' esté en el mismo directorio que app.py
try:
    model = joblib.load('best_decision_tree_model.pkl')
except FileNotFoundError:
    st.error("Error: El archivo del modelo 'best_decision_tree_model.pkl' no se encontró.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Título del Dashboard
st.title("🧠 Predicción de Probabilidad de Alzheimer")

st.write("Complete el siguiente formulario con la información del paciente para obtener una predicción del riesgo de Alzheimer.")

# Sección: Detalles Demográficos
with st.expander("📊 Detalles Demográficos", expanded=True):
    age = st.number_input("Edad (60 a 90 años)", min_value=60, max_value=90, value=70)
    gender = st.radio("Género", ["Masculino", "Femenino"])
    ethnicity = st.selectbox("Etnicidad", ["Caucásico", "Afroamericano", "Asiático", "Otros"])
    education = st.selectbox("Nivel de Educación", ["Ninguno", "Escuela secundaria", "Licenciatura", "Superior"])

# Sección: Estilo de Vida
with st.expander("🏃 Factores del Estilo de Vida", expanded=True):
    bmi = st.number_input("IMC (15.0 a 40.0)", min_value=15.0, max_value=40.0, value=25.0, step=0.1)
    smoking = st.radio("¿Fuma?", ["No", "Sí"])
    alcohol = st.number_input("Consumo de alcohol semanal (0 a 20 unidades)", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
    physical = st.number_input("Actividad física semanal (0 a 10 horas)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
    DietQuality = st.slider("Calidad de la dieta (0 a 10)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sleep_quality = st.slider("Calidad del sueño (4 a 10)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)

# Sección: Historial Médico
with st.expander("🏥 Historial Médico", expanded=True):
    family_history = st.radio("Antecedentes familiares de Alzheimer", ["No", "Sí"])
    cardiovascular = st.radio("¿Tiene enfermedad cardiovascular?", ["No", "Sí"])
    diabetes = st.radio("¿Tiene diabetes?", ["No", "Sí"])
    depression = st.radio("¿Tiene depresión?", ["No", "Sí"])
    head_injury = st.radio("¿Ha sufrido lesión en la cabeza?", ["No", "Sí"])
    hypertension = st.radio("¿Tiene hipertensión?", ["No", "Sí"])

# Sección: Mediciones Clínicas
with st.expander("🩺 Mediciones Clínicas", expanded=True):
    systolic = st.number_input("Presión arterial sistólica (90 a 180 mmHg)", min_value=90, max_value=180, value=120)
    diastolic = st.number_input("Presión arterial diastólica (60 a 120 mmHg)", min_value=60, max_value=120, value=80)
    chol_total = st.number_input("Colesterol total (150 a 300 mg/dL)", min_value=150.0, max_value=300.0, value=200.0, step=1.0)
    chol_ldl = st.number_input("Colesterol LDL (50 a 200 mg/dL)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)
    chol_hdl = st.number_input("Colesterol HDL (20 a 100 mg/dL)", min_value=20.0, max_value=100.0, value=50.0, step=1.0)
    chol_trig = st.number_input("Triglicéridos (50 a 400 mg/dL)", min_value=50.0, max_value=400.0, value=150.0, step=1.0)

# Sección: Evaluaciones cognitivas y funcionales
with st.expander("🧠 Evaluaciones Cognitivas y Funcionales", expanded=True):
    mmse = st.slider("Puntuación MMSE (0 a 30)", min_value=0.0, max_value=30.0, value=25.0, step=1.0)
    func_assess = st.slider("Evaluación funcional (0 a 10)", min_value=0.0, max_value=10.0, value=8.0, step=1.0)
    memory_complaints = st.radio("¿Tiene quejas de memoria?", ["No", "Sí"])
    behavioral_problems = st.radio("¿Tiene problemas de conducta?", ["No", "Sí"])
    adl = st.slider("Puntuación AVD (0 a 10)", min_value=0.0, max_value=10.0, value=9.0, step=1.0)

# Sección: Síntomas
with st.expander("🧩 Síntomas", expanded=True):
    confusion = st.radio("¿Tiene confusión?", ["No", "Sí"])
    disorientation = st.radio("¿Tiene desorientación?", ["No", "Sí"])
    personality_changes = st.radio("¿Tiene cambios de personalidad?", ["No", "Sí"])
    task_difficulty = st.radio("¿Tiene dificultad para completar tareas?", ["No", "Sí"])
    forgetfulness = st.radio("¿Tiene olvidos frecuentes?", ["No", "Sí"])

# Mapas de codificación
def codificar(valor, mapa):
    return mapa.get(valor, 0) # Devuelve 0 si el valor no está en el mapa (puedes ajustar esto si es necesario)

# Diccionarios para codificación
genero_map = {"Masculino": 0, "Femenino": 1}
etnia_map = {"Caucásico": 0, "Afroamericano": 1, "Asiático": 2, "Otros": 3}
educacion_map = {"Ninguno": 0, "Escuela secundaria": 1, "Licenciatura": 2, "Superior": 3}
binario_map = {"No": 0, "Sí": 1}

# Botón para generar predicción
if st.button("🔍 Diagnóstico de Alzheimer predicho"):
    # Crear diccionario con los datos ingresados
    input_dict = {
        'Age': age,
        'Gender': codificar(gender, genero_map),
        'Ethnicity': codificar(ethnicity, etnia_map),
        'EducationLevel': codificar(education, educacion_map),
        'BMI': bmi,
        'Smoking': codificar(smoking, binario_map),
        'AlcoholConsumption': alcohol,
        'PhysicalActivity': physical,
        'DietQuality': DietQuality,
        'SleepQuality': sleep_quality,
        'FamilyHistoryAlzheimers': codificar(family_history, binario_map),
        'CardiovascularDisease': codificar(cardiovascular, binario_map),
        'Diabetes': codificar(diabetes, binario_map),
        'Depression': codificar(depression, binario_map),
        'HeadInjury': codificar(head_injury, binario_map),
        'Hypertension': codificar(hypertension, binario_map),
        'SystolicBP': systolic,
        'DiastolicBP': diastolic,
        'CholesterolTotal': chol_total,
        'CholesterolLDL': chol_ldl,
        'CholesterolHDL': chol_hdl,
        'CholesterolTriglycerides': chol_trig,
        'MMSE': mmse,
        'FunctionalAssessment': func_assess,
        'MemoryComplaints': codificar(memory_complaints, binario_map),
        'BehavioralProblems': codificar(behavioral_problems, binario_map),
        'ADL': adl,
        'Confusion': codificar(confusion, binario_map),
        'Disorientation': codificar(disorientation, binario_map),
        'PersonalityChanges': codificar(personality_changes, binario_map),
        'DifficultyCompletingTasks': codificar(task_difficulty, binario_map),
        'Forgetfulness': codificar(forgetfulness, binario_map)
    }

    # Convertir a DataFrame.
    # Es CRUCIAL que el orden y los nombres de las columnas coincidan exactamente
    # con las que el modelo fue entrenado (después del preprocesamiento).
    # Si tu modelo espera un orden específico, asegúrate de que input_df lo respete.
    # Por ejemplo, si tienes una lista de columnas:
    # FEATURE_COLUMNS = ['Age', 'Gender', ..., 'Forgetfulness'] # El orden exacto
    # input_df = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0] # Para obtener probabilidades

        if prediction == 1:
            st.error(f"Predicción: Alzheimer probable")
            st.metric(label="Probabilidad de Alzheimer", value=f"{probability[1]*100:.2f}%")
        else:
            st.success(f"Predicción: Sin Alzheimer (o bajo riesgo)")
            st.metric(label="Probabilidad de No Alzheimer", value=f"{probability[0]*100:.2f}%")

        # Opcional: Mostrar más detalles de las probabilidades
        # st.write("Probabilidades detalladas:", {
        # 'No Alzheimer': f"{probability[0]*100:.2f}%",
        # 'Alzheimer': f"{probability[1]*100:.2f}%"
        # })

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
        st.write("Asegúrate de que los datos de entrada son correctos y el modelo está cargado.")