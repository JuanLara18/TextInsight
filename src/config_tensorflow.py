import os
import tensorflow as tf

# Desactivar los logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = todos los mensajes, 1 = filtros de información, 2 = filtros de advertencias, 3 = filtros de errores

# Desactivar los custom operations de oneDNN para evitar resultados numéricos ligeramente diferentes
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suprimir advertencias de deprecación
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)