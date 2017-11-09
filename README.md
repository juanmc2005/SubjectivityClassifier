## Tesina de Grado - Coria Juan Manuel

### Requisitos Previos

- Python 3+
- Pip 3
- Java 8 con las variables de entorno configuradas correctamente

### Instalar Dependencias

- Posicionarse en la carpeta `src`
- Ejecutar `pip3 install -r requirements.txt`
- Ejecutar `python3 install_nltk_modules.py`

### Ejemplo 1

`python3 example1.py`

Al finalizar la ejecución se almacena en el directorio `svm_dump` la información del modelo óptimo y las particiones de la base de datos, a su vez también se puede encontrar un archivo CSV con los resultados generados en cada iteración de la búsqueda.

**Nota:** La ejecución de este ejemplo puede demorar bastante (~1h)

### Ejemplo 2

`python3 example2.py`

En este ejemplo se carga el SVM óptimo descrito en el informe, y se muestran los resultados de su evaluación en la pantalla.

### Notas

- El directorio `stanford` contiene todos los archivos necesarios para ejecutar el POS tagger de Stanford
- Los directorios `nn_cross` y `svm_cross` contienen toda la información relacionada a los modelos óptimos hallados
- El directorio `bd` contiene la base de datos en forma de texto plano y en su forma vectorizada