import socket
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import time

# Define UserBehavior dataclass
@dataclass
class UserBehavior:
    typing_speed: float  # WPM
    keystroke_intervals: List[float]  # milliseconds
    mouse_movement: List[Tuple[int, int]]  # x, y coordinates
    mouse_velocity: float  # pixels/second
    idle_time: float  # seconds

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

def start_client(server_host='127.0.0.1', server_port=5555, user_id="user_1"):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_host, server_port))
    print(f"Connected to server at {server_host}:{server_port}")

    # Register
    client.send(pickle.dumps({"action": "register", "user_id": user_id}))
    response = pickle.loads(client.recv(4096))
    print(response)

    # Train
    client.send(pickle.dumps({"action": "train"}))
    response = pickle.loads(client.recv(4096))
    print(response)

    start_time = time.time()
    while time.time() - start_time < 60:  # 1 minute of training
        user_behavior = generate_sample_behavior(is_legitimate=True)
        client.send(pickle.dumps(user_behavior))

    client.send(pickle.dumps({"action": "stop_train"}))
    response = pickle.loads(client.recv(4096))
    print(response)

    # Authenticate
    test_behavior = generate_sample_behavior(is_legitimate=True)
    client.send(pickle.dumps({"action": "authenticate", "data": test_behavior}))
    response = pickle.loads(client.recv(4096))
    print(response)

    client.close()

if __name__ == "__main__":
    start_client()
