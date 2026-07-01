{
  "propuesta_titulo": "Predicción de band gap en óxidos metálicos mediante Machine Learning para aplicaciones en celdas solares",
  "herramientas_seleccionadas": [
    "U1_modelado_atomistico",
    "U3_ml_clasico",
    "U3_redes_neuronales",
    "U5_agentes_langchain",
    "U5_rag_memoria",
    "U6_api_fastapi"
  ],
  "pipeline": [
    {
      "etapa": "Adquisición de datos",
      "descripcion": "Descarga de datos estructurales y band gaps de óxidos metálicos usando Materials Project API",
      "herramienta": "U1_modelado_atomistico"
    },
    {
      "etapa": "Extracción de descriptores",
      "descripcion": "Generación de características químicas y geométricas (electronegatividad, volumen, coordinación) con pymatgen y ASE",
      "herramienta": "U1_modelado_atomistico"
    },
    {
      "etapa": "Entrenamiento de modelos de ML",
      "descripcion": "Ajuste de modelos clásicos de ensamble (Random Forest, Gradient Boosting) y redes neuronales MLP para predecir el bandgap",
      "herramienta": "U3_ml_clasico"
    },
    {
      "etapa": "RAG de literatura de soporte",
      "descripcion": "Búsqueda semántica en base de datos de papers vectorizada para validar experimentalmente los bandgaps predichos",
      "herramienta": "U5_rag_memoria"
    },
    {
      "etapa": "Despliegue de API REST",
      "descripcion": "Creación de un endpoint FastAPI para exponer el modelo y permitir screening de nuevos óxidos metálicos",
      "herramienta": "U6_api_fastapi"
    }
  ],
  "pipeline_completo": true
}