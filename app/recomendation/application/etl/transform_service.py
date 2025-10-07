import pandas as pd                     # para manejar datos tipo tabla (DataFrame)
import os                               # para manejar rutas y carpetas del sistema
import marshmallow as ma                # para validar datos de forma estructurada
from datetime import datetime           # para manejar fechas con precisión


# 🧱 Esquema de validación de datos (con alias 'ma')
# Marshmallow nos ayuda a garantizar que cada registro tenga los tipos correctos.
class CompraSchema(ma.Schema):
    id_compra = ma.fields.Int(required=True)       # entero obligatorio
    cliente = ma.fields.Str(required=True)         # cadena obligatoria
    producto = ma.fields.Str(required=True)        # cadena obligatoria
    categoria = ma.fields.Str(required=True)       # cadena obligatoria
    precio = ma.fields.Float(required=True)        # número decimal obligatorio
    fecha_compra = ma.fields.Date(required=True)   # fecha obligatoria (YYYY-MM-DD)

    # Método para convertir fechas si vienen en texto
    def convertir_fecha(self, data):
        """Convierte la fecha a tipo date si viene como cadena."""
        if isinstance(data.get("fecha_compra"), str):
            try:
                data["fecha_compra"] = datetime.strptime(data["fecha_compra"], "%Y-%m-%d").date()
            except ValueError:
                data["fecha_compra"] = None
        return data


def transform_data():
    """
    Limpia, transforma y valida los datos del archivo compras_raw.csv.
    Incluye:
    - Limpieza general con Pandas (duplicados, nulos, formato)
    - Validación de tipos y estructura con Marshmallow
    - Exportación final en /data/clean/compras_clean.csv (separador ';')
    """

    # 📂 1️⃣ Definir rutas de entrada y salida
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ruta_raw = os.path.join(base_dir, "infrastructure", "data", "raw", "compras_raw.csv")
    ruta_clean = os.path.join(base_dir, "infrastructure", "data", "clean")
    os.makedirs(ruta_clean, exist_ok=True)
    ruta_salida = os.path.join(ruta_clean, "compras_clean.csv")

    # 📥 2️⃣ Lectura segura del CSV crudo
    # Forzamos el separador coma (',') porque es el formato generado en el extract.
    df = pd.read_csv(ruta_raw, encoding="utf-8-sig", sep=",")
    print("✅ Archivo leído correctamente")
    print("Columnas detectadas:", df.columns.tolist())

    # 🧹 3️⃣ Limpieza básica
    #          Quita espacios  minusculas
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df = df.drop_duplicates()   # → elimina cualquier fila repetida exacta.
    df = df.dropna(how="all")   # → elimina filas donde todas las columnas están vacías (nulos o NaN).

    # eliminar duplicados solo por cliente o producto, se puede hacer más específico:
    # df = df.drop_duplicates(subset=["cliente", "producto"])

    # 🔢 4️⃣ Conversión de tipos inicial (antes de validar)
    # Si hay errores de tipo, Pandas los convierte a NaN para filtrarlos después.
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    df["fecha_compra"] = pd.to_datetime(df["fecha_compra"], errors="coerce")

    # Eliminamos filas sin cliente o sin fecha (datos esenciales)
    df = df.dropna(subset=["cliente", "fecha_compra"], how="any")

    # 🧠 5️⃣ Validación de estructura con Marshmallow
    schema = CompraSchema()
    registros_validos = []
    registros_invalidos = []

    for i, row in df.iterrows():
        record = row.to_dict()
        record = schema.convertir_fecha(record)
        errors = schema.validate(record)
        if errors:
            registros_invalidos.append({"fila": i, "errores": errors})
        else:
            registros_validos.append(record)

    # 📋 Mostrar los errores encontrados (solo los primeros 5)
    if registros_invalidos:
        print(f"⚠️ Se encontraron {len(registros_invalidos)} registros inválidos:")
        for err in registros_invalidos[:5]:
            print(f" - Fila {err['fila']}: {err['errores']}")

    # 🧮 6️⃣ Crear nuevo DataFrame con solo los registros válidos
    df_limpio = pd.DataFrame(registros_validos)

    # 💾 7️⃣ Guardar el archivo limpio en formato compatible con Excel (';')
    df_limpio.to_csv(ruta_salida, index=False, encoding="utf-8-sig", sep=";")

    print(f"🧼 Archivo limpio y validado guardado en: {ruta_salida}")
    print(f"Registros válidos: {len(df_limpio)} / {len(df)}")

    return ruta_salida
