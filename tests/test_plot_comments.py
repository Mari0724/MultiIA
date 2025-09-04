import pytest
from app.nlp.application.plot_comments import generar_graficas

def test_generar_graficas_no_guardar():
    """
    Genera gráfica SIN guardar archivo.
    """
    data = generar_graficas(save_plot=False)
    assert "conteo" in data
    assert isinstance(data["conteo"], dict)
    # aseguramos que no devuelva error si hay comentarios
    assert not ("error" in data and data["error"])

def test_generar_graficas_guardar(tmp_path):
    """
    Genera gráfica y guarda archivo en un directorio temporal.
    """
    # ⚡️ truco: cambiamos el PLOTS_DIR solo para este test
    import app.nlp.application.plot_comments as plots
    plots.PLOTS_DIR = tmp_path

    data = generar_graficas(save_plot=True)
    assert "conteo" in data
    archivos = list(tmp_path.glob("*.png"))
    assert len(archivos) > 0  # se generó una gráfica
