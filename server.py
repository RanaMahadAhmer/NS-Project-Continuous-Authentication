import socket
import threading
import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Tuple


# Define UserBehavior dataclass
@dataclass
class UserBehavior:
    typing_speed: float  # WPM
    keystroke_intervals: List[float]  # milliseconds
    mouse_movement: List[Tuple[int, int]]  # x, y coordinates
    mouse_velocity: float  # pixels/second
    idle_time: float  # seconds


# ContinuousAuthenticator class
class ContinuousAuthenticator:
    def __init__(self, security_threshold: float = 0.85):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.security_threshold = security_threshold
        self.trained = False

    def extract_features(self, behavior: UserBehavior) -> np.ndarray:
        features = [
            behavior.typing_speed,
            np.mean(behavior.keystroke_intervals),
            np.std(behavior.keystroke_intervals),
            behavior.mouse_velocity,
            behavior.idle_time
        ]
        if behavior.mouse_movement:
            x_coords, y_coords = zip(*behavior.mouse_movement)
            features.extend([
                np.mean(x_coords),
                np.std(x_coords),
                np.mean(y_coords),
                np.std(y_coords)
            ])
        else:
            features.extend([0, 0, 0, 0])
        return np.array(features).reshape(1, -1)

    def train(self, user_data: List[UserBehavior]):
        if not user_data:
            raise ValueError("No training data provided.")
        X, y = [], []
        for behavior in user_data:
            features = self.extract_features(behavior)
            X.append(features.flatten())
            y.append(1)  # Mark all training data as legitimate
        X = np.array(X)
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.trained = True

    def authenticate(self, current_behavior: UserBehavior) -> Tuple[bool, float]:
        if not self.trained:
            raise RuntimeError("Model has not been trained.")
        features = self.extract_features(current_behavior)
        scaled_features = self.scaler.transform(features)
        auth_prob = self.model.predict_proba(scaled_features)[0][1]
        return auth_prob >= self.security_threshold, auth_prob


# Server function
clients_data = {}  # Store client-specific training data


def handle_client(client_socket, address):
    global clients_data
    user_id = None
    authenticator = ContinuousAuthenticator()

    try:
        while True:
            # Receive serialized data
            data = client_socket.recv(4096)
            if not data:
                break
            message = pickle.loads(data)

            if message["action"] == "register":
                user_id = message["user_id"]
                if user_id not in clients_data:
                    clients_data[user_id] = []
                client_socket.send(pickle.dumps({"status": "registered"}))

            elif message["action"] == "train":
                if not user_id:
                    client_socket.send(pickle.dumps({"error": "Register first."}))
                    continue
                client_socket.send(pickle.dumps({"status": "training started"}))
                start_time = time.time()
                while time.time() - start_time < 60:  # 1 minute
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    user_behavior = pickle.loads(data)
                    clients_data[user_id].append(user_behavior)
                client_socket.send(pickle.dumps({"status": "training stopped"}))

            elif message["action"] == "stop_train":
                if not user_id:
                    client_socket.send(pickle.dumps({"error": "Register first."}))
                    continue
                authenticator.train(clients_data[user_id])
                client_socket.send(pickle.dumps({"status": "model trained"}))

            elif message["action"] == "authenticate":
                if not user_id:
                    client_socket.send(pickle.dumps({"error": "Register first."}))
                    continue
                user_behavior = message["data"]
                is_auth, prob = authenticator.authenticate(user_behavior)
                client_socket.send(pickle.dumps({"authenticated": is_auth, "probability": prob}))

    except Exception as e:
        print(f"Error handling client {address}: {e}")
    finally:
        client_socket.close()


def start_server(host='127.0.0.1', port=5555):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")
    while True:
        client_socket, address = server.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_socket, address))
        client_thread.start()


if __name__ == "__main__":
    start_server()
