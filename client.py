import socket
import pickle
import time

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class UserBehavior:
    typing_speed: float
    keystroke_intervals: List[float]
    mouse_movement: List[Tuple[int, int]]
    mouse_velocity: float
    idle_time: float


def generate_sample_behavior(is_legitimate: bool = True) -> UserBehavior:
    base_typing_speed = 65 if is_legitimate else 45
    base_keystroke_interval = 200 if is_legitimate else 300
    return UserBehavior(
        typing_speed=base_typing_speed + np.random.normal(0, 5),
        keystroke_intervals=[base_keystroke_interval + np.random.normal(0, 20) for _ in range(10)],
        mouse_movement=[(int(np.random.normal(500, 100)), int(np.random.normal(500, 100))) for _ in range(5)],
        mouse_velocity=np.random.normal(400 if is_legitimate else 300, 50),
        idle_time=np.random.normal(5 if is_legitimate else 10, 1)
    )


def start_client(server_host='127.0.0.1', server_port=5555):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_host, server_port))
    print(f"Connected to server at {server_host}:{server_port}")

    try:
        # Device Registration
        device_info = {"device_name": "Client1", "os": "Windows", "version": "1.0"}
        client.send(pickle.dumps({"type": "register", "device_info": device_info}))
        response = pickle.loads(client.recv(4096))
        token = response["token"]
        print(f"Device registered. Token: {token}")

        # Authenticate Device
        client.send(pickle.dumps({"type": "authenticate", "token": token, "device_info": device_info}))
        auth_response = pickle.loads(client.recv(4096))
        print(f"Authentication Status: {auth_response['message']}")

        # Training the model
        print("Training the model for 1 minute...")
        behaviors = [generate_sample_behavior(True) for _ in range(5)]  # Random behaviors for training
        impostor_behaviors = [generate_sample_behavior(False) for _ in range(5)]
        client.send(pickle.dumps({"type": "train", "behaviors": behaviors, "impostor_behaviors": impostor_behaviors}))
        train_response = pickle.loads(client.recv(4096))
        print(train_response["message"])

        # Testing the model after training
        print("Testing the model with altered behavior...Actual User")
        altered_behavior = generate_sample_behavior(True)  # Altered behavior for testing
        client.send(pickle.dumps({"type": "test", "current_behavior": altered_behavior}))
        test_response = pickle.loads(client.recv(4096))
        print(f"Authenticated: {test_response['authenticated']}, Probability: {test_response['probability']}")

        # Testing the model after training
        print("Testing the model with altered behavior...Imposter")
        altered_behavior = generate_sample_behavior(False)  # Altered behavior for testing
        client.send(pickle.dumps({"type": "test", "current_behavior": altered_behavior}))
        test_response = pickle.loads(client.recv(4096))
        print(f"Authenticated: {test_response['authenticated']}, Probability: {test_response['probability']}")

    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        client.close()


if __name__ == "__main__":
    start_client()
