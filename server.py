import socket
import threading
import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List, Tuple, Dict
import hashlib
import hmac
import secrets


@dataclass
class UserBehavior:
    typing_speed: float
    keystroke_intervals: List[float]
    mouse_movement: List[Tuple[int, int]]
    mouse_velocity: float
    idle_time: float


class ContinuousAuthenticator:
    def __init__(self, security_threshold: float = 0.85):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        self.security_threshold = security_threshold
        self.session_key = secrets.token_hex(32)
        self.trusted_devices = {}

    def generate_device_signature(self, device_info: Dict) -> str:
        """Generate unique device signature using device characteristics"""
        device_str = ''.join(str(v) for v in device_info.values())
        return hashlib.sha256(device_str.encode()).hexdigest()

    def register_device(self, device_info: Dict) -> str:
        """Register a trusted device"""
        signature = self.generate_device_signature(device_info)
        token = secrets.token_hex(16)
        self.trusted_devices[signature] = {
            'token': token,
            'timestamp': time.time()
        }
        return token

    def verify_device(self, device_info: Dict, token: str) -> bool:
        """Verify if device is trusted"""
        signature = self.generate_device_signature(device_info)
        if signature in self.trusted_devices:
            stored_token = self.trusted_devices[signature]['token']
            return hmac.compare_digest(stored_token, token)
        return False

    def extract_features(self, behavior: UserBehavior) -> np.ndarray:
        """Extract features from user behavior"""
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

    def train(self, legitimate_behaviors: List[UserBehavior], impostor_behaviors: List[UserBehavior]):
        """Train the authentication model"""
        X = []
        y = []

        # Process legitimate user behaviors
        for behavior in legitimate_behaviors:
            features = self.extract_features(behavior)
            X.append(features.flatten())
            y.append(1)

        # Process impostor behaviors
        for behavior in impostor_behaviors:
            features = self.extract_features(behavior)
            X.append(features.flatten())
            y.append(0)

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)

    def authenticate(self, current_behavior: UserBehavior) -> Tuple[bool, float]:
        """Authenticate user based on current behavior"""
        features = self.extract_features(current_behavior)
        scaled_features = self.scaler.transform(features)

        # Get authentication probability
        auth_prob = self.model.predict_proba(scaled_features)[0][1]

        # Authenticate if probability exceeds threshold
        is_authenticated = auth_prob >= self.security_threshold

        return is_authenticated, auth_prob


# Server handling each client
def handle_client(client_socket):
    device_info = {}  # To store device registration info
    authenticator = ContinuousAuthenticator(security_threshold=0.85)

    while True:
        try:
            data = client_socket.recv(4096)
            if not data:
                break

            message = pickle.loads(data)
            
            if message["type"] == "register":
                # Register device
                token = authenticator.register_device(message["device_info"])
                client_socket.send(pickle.dumps({"status": "registered", "token": token}))
                # print(message)

            elif message["type"] == "authenticate":
                # Authenticate device using token
                token = message["token"]
                device_info = message["device_info"]
                if authenticator.verify_device(device_info, token):
                    client_socket.send(pickle.dumps({"status": "authenticated", "message": "Device authenticated"}))
                else:
                    client_socket.send(pickle.dumps({"status": "failed", "message": "Device authentication failed"}))
                # print(message)


            elif message["type"] == "train":

                # Train the model
                behaviors = message["behaviors"]
                impostor_behaviors = message["impostor_behaviors"]
                authenticator.train(behaviors, impostor_behaviors)
                # print(message)
                client_socket.send(
                    pickle.dumps({"status": "training_complete", "message": "Model trained successfully"}))
                # print(message)

            elif message["type"] == "test":
                # Test the model with new behavior
                current_behavior = message["current_behavior"]
                is_auth, prob = authenticator.authenticate(current_behavior)
                client_socket.send(pickle.dumps({"authenticated": is_auth, "probability": prob}))

        except Exception as e:
            print(f"Error: {e}")
            break

    client_socket.close()


def start_server(host='127.0.0.1', port=5555):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")
    while True:
        client_socket, _ = server.accept()
        client_thread = threading.Thread(target=handle_client, args=(client_socket,))
        client_thread.start()


if __name__ == "__main__":
    start_server()
