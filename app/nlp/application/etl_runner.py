from app.nlp.application.etl_extract import ETLExtractor
from app.nlp.application.etl_transform import ETLTransformer
from app.nlp.application.etl_load import load_data
from app.nlp.infrastructure.db import DBConnection

def run_etl():
    print("🚀 Iniciando proceso ETL...")

    db = DBConnection()

    # Fases del ETL
    extractor = ETLExtractor(db)
    extracted_data = extractor.extract()

    transformer = ETLTransformer()
    transformed_data = transformer.transform(extracted_data)

    # 💾 Carga de datos (pasamos los datos al método)
    load_data(transformed_data)

    print("✅ ETL completada exitosamente.")


if __name__ == "__main__":
    run_etl()
