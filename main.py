import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass
import hashlib
import hmac
import secrets


@dataclass
class UserBehavior:
    typing_speed: float  # WPM
    keystroke_intervals: List[float]  # milliseconds
    mouse_movement: List[Tuple[int, int]]  # x, y coordinates
    mouse_velocity: float  # pixels/second
    idle_time: float  # seconds


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

        # Add mouse movement patterns
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

    def train(self, legitimate_behaviors: List[UserBehavior],
              impostor_behaviors: List[UserBehavior]):
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


# Test implementation against passive attack
def test_passive_attack(authenticator: ContinuousAuthenticator,
                        legitimate_behavior: UserBehavior,
                        attack_behaviors: List[UserBehavior]) -> Dict:
    """Test system against passive mimicry attack"""
    results = {
        'legitimate_auth': None,
        'attack_success_rate': 0,
        'attack_attempts': len(attack_behaviors),
        'average_attack_probability': 0
    }

    # Test legitimate authentication
    is_auth, prob = authenticator.authenticate(legitimate_behavior)
    results['legitimate_auth'] = is_auth

    # Test attack behaviors
    successful_attacks = 0
    attack_probs = []

    for attack_behavior in attack_behaviors:
        is_auth, prob = authenticator.authenticate(attack_behavior)
        if is_auth:
            successful_attacks += 1
        attack_probs.append(prob)

    results['attack_success_rate'] = successful_attacks / len(attack_behaviors)
    results['average_attack_probability'] = np.mean(attack_probs)

    return results


# Example usage and testing
def generate_sample_behavior(is_legitimate: bool = True) -> UserBehavior:
    """Generate sample behavior data for testing"""
    base_typing_speed = 65 if is_legitimate else 45
    base_keystroke_interval = 200 if is_legitimate else 300

    return UserBehavior(
        typing_speed=base_typing_speed + np.random.normal(0, 5),
        keystroke_intervals=[base_keystroke_interval + np.random.normal(0, 20)
                             for _ in range(10)],
        mouse_movement=[(int(np.random.normal(500, 100)),
                         int(np.random.normal(500, 100)))
                        for _ in range(5)],
        mouse_velocity=np.random.normal(400 if is_legitimate else 300, 50),
        idle_time=np.random.normal(5 if is_legitimate else 10, 1)
    )


# Main execution
if __name__ == "__main__":
    # Initialize authenticator
    authenticator = ContinuousAuthenticator(security_threshold=0.85)

    # Generate training data
    legitimate_behaviors = [generate_sample_behavior(True) for _ in range(100)]
    impostor_behaviors = [generate_sample_behavior(False) for _ in range(100)]

    # Train the model
    authenticator.train(legitimate_behaviors, impostor_behaviors)

    # Generate test data
    legitimate_test = generate_sample_behavior(True)
    attack_behaviors = [generate_sample_behavior(False) for _ in range(20)]

    # Test against passive attack
    results = test_passive_attack(authenticator, legitimate_test, attack_behaviors)

    # Print results
    print("\nTest Results:")
    print(f"Legitimate Authentication Successful: {results['legitimate_auth']}")
    print(f"Attack Success Rate: {results['attack_success_rate'] * 100:.2f}%")
    print(f"Average Attack Probability: {results['average_attack_probability']:.3f}")
