
# --- 1. KONFIGURASI UTAMA ---
DLIB_LANDMARK_MODEL_PATH = "./shape_predictor_68_face_landmarks.dat"
FACE_EMBEDDINGS_PATH = "./face_embeddings_tf.pkl"
FACE_MODEL_PATH = "./face_embedding_model.h5"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "iot/commands/eyeblink"
MQTT_MESSAGE = "TOGGLE"
MQTT_TOPIC_LIGHT = "iot/commands/light"