import pandas as pd                     # para manejar datos en forma de tabla (DataFrame)
import random                           # para elegir valores aleatorios (producto, categoría, precio)
from faker import Faker                 # para generar datos falsos realistas (nombres, fechas, etc.)
from datetime import datetime, timedelta
import os                               # para manejar rutas y carpetas del sistema

fake = Faker('es_CO')  # datos realistas en español colombiano

# 🏷️ Catálogo de productos organizados por categoría
PRODUCTOS = {
    "Alimentos": [
        "Arroz Roa", "Fríjoles Diana", "Lentejas La Constancia", "Aceite Premier",
        "Azúcar Manuelita", "Sal Refisal", "Atún Van Camps", "Harina Pan"
    ],
    "Limpieza": [
        "Jabón Rey", "Detergente Ariel", "Suavitel", "Cloro Clorox",
        "Papel Higiénico Familia", "Lavaloza Axion", "Desinfectante Fabuloso", "Toallas de Cocina Scott"
    ],
    "Bebidas": [
        "Coca-Cola", "Agua Cristal", "Jugo Hit", "Leche Alquería",
        "Jugo de Guayaba del Valle", "Gaseosa Postobón Manzana", "Agua Brisa con gas", "Té Lipton Durazno"
    ],
    "Granos": [
        "Maíz Blanco", "Garbanzo", "Avena", "Lentejas Ramo",
        "Fríjol Cargamanto", "Maíz Pira", "Trigo Integral", "Arveja Verde"
    ]
}


def generar_dataset(cantidad=30000):
    """
    Genera un dataset de compras sintéticas y lo guarda como CSV.
    Incluye:
    - Cliente (nombre aleatorio)
    - Producto y categoría
    - Precio y fecha de compra
    """

    data = []

    for i in range(1, cantidad + 1):
        categoria = random.choice(list(PRODUCTOS.keys()))  # elige una categoría al azar
        producto = random.choice(PRODUCTOS[categoria])     # elige un producto dentro de esa categoría
        cliente = fake.name()                              # genera un nombre de cliente falso
        precio = round(random.uniform(2000, 30000), 2)     # genera un precio aleatorio entre 2.000 y 30.000
        fecha_compra = fake.date_between(start_date="-1y", end_date="today")  # fecha entre hace un año y hoy

        data.append({
            "id_compra": i,
            "cliente": cliente,
            "producto": producto,
            "categoria": categoria,
            "precio": precio,
            "fecha_compra": fecha_compra
        })

    # 📂 Crear carpeta si no existe
    base_dir = os.path.dirname(os.path.dirname(__file__))  # sube dos niveles (para seguir la estructura del proyecto)
    ruta_carpeta = os.path.join(base_dir, "infrastructure", "data", "raw")
    os.makedirs(ruta_carpeta, exist_ok=True)

    ruta = os.path.join(ruta_carpeta, "compras_raw.csv")

    # 💾 Guardar archivo generado
    df = pd.DataFrame(data)
    df.to_csv(ruta, index=False, encoding="utf-8-sig")

    # 📊 Confirmación y ejemplo de columnas
    print(f"✅ Dataset generado en {ruta}")
    print(f"Total de registros generados: {len(df)}")
    print(f"Columnas: {list(df.columns)}")

    #eliminar duplicados solo por cliente o producto, se puede hacer más específico:
    #df = df.drop_duplicates(subset=["cliente", "producto"])

    return ruta
