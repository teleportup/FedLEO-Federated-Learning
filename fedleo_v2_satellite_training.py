import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

print(f"🚀 TensorFlow version: {tf.__version__}\n")

class SatelliteV2:
    """Спутник v2.0 - обучение полностью на орбите"""
    def __init__(self, sat_id, model, local_data, local_labels):
        self.sat_id = sat_id
        self.model = model
        self.local_data = local_data
        self.local_labels = local_labels
        self.local_dataset_size = len(local_data)
        self.training_history = {'loss': [], 'accuracy': [], 'time': []}
        self.orbital_altitude = 400 + sat_id * 50
        self.total_training_time = 0
        self.weights_sent = 0
        print(f"🛰️  Спутник {sat_id} инициализирован | Данные: {self.local_dataset_size} | Высота: {self.orbital_altitude} км")
    
    def receive_weights(self, weights):
        """Получить веса с Земли"""
        self.model.set_weights(weights)
    
    def train_on_satellite(self, epochs=1, lr=0.01):
        """⭐ ГЛАВНОЕ: Локальное обучение полностью на спутнике"""
        start = time.time()
        
        self.model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.model.fit(
            self.local_data, self.local_labels,
            epochs=epochs, verbose=0, batch_size=32, shuffle=True
        )
        
        train_time = time.time() - start
        self.total_training_time += train_time
        
        loss = history.history['loss'][-1]
        acc = history.history['accuracy'][-1]
        
        self.training_history['loss'].append(loss)
        self.training_history['accuracy'].append(acc)
        self.training_history['time'].append(train_time)
        
        return {'loss': loss, 'accuracy': acc, 'time': train_time}
    
    def send_weights(self):
        """Отправить веса на Землю (только веса, 200KB!)"""
        self.weights_sent += 1
        return self.model.get_weights()
    
    def get_dataset_size(self):
        return self.local_dataset_size


class GroundStationV2:
    """Наземная станция v2.0 - только координирует и усредняет"""
    def __init__(self, global_model):
        self.global_model = global_model
        self.global_weights = [w.copy() for w in global_model.get_weights()]
        self.history = {'round': [], 'avg_loss': [], 'avg_acc': []}
        print("🌍 Наземная станция инициализирована\n")
    
    def broadcast_weights(self, satellites):
        """Отправить веса на спутники"""
        for sat in satellites:
            sat.receive_weights(self.global_weights)
    
    def aggregate_weights(self, satellites):
        """Усреднить веса со спутников"""
        all_weights = [sat.send_weights() for sat in satellites]
        all_sizes = [sat.get_dataset_size() for sat in satellites]
        total_size = sum(all_sizes)
        
        aggregated = []
        for layer_idx in range(len(all_weights[0])):
            weighted = None
            for sat_idx in range(len(satellites)):
                coeff = all_sizes[sat_idx] / total_size
                if weighted is None:
                    weighted = coeff * all_weights[sat_idx][layer_idx]
                else:
                    weighted += coeff * all_weights[sat_idx][layer_idx]
            aggregated.append(weighted)
        
        self.global_model.set_weights(aggregated)
        self.global_weights = [w.copy() for w in aggregated]


def fedleo_v2_training(satellites, ground_station, num_rounds=2, epochs=1):
    """Главный алгоритм FedLEO v2.0"""
    print("\n" + "="*80)
    print("🚀 FedLEO v2.0: ФЕДЕРАТИВНОЕ ОБУЧЕНИЕ (ОБУЧЕНИЕ НА СПУТНИКАХ)")
    print("="*80)
    print(f"Спутников: {len(satellites)} | Раундов: {num_rounds}")
    print("✨ Ключевая особенность: Данные ОСТАЮТСЯ на спутниках!")
    print("="*80 + "\n")
    
    for round_num in range(num_rounds):
        print(f"\n┌{'─'*78}┐")
        print(f"│ 📡 РАУНД {round_num + 1}/{num_rounds}")
        print(f"└{'─'*78}┘\n")
        
        # 1. BROADCAST
        print(f"1️⃣  BROADCAST (Земля → Спутники)")
        ground_station.broadcast_weights(satellites)
        print(f"   ✓ Веса отправлены на {len(satellites)} спутников\n")
        
        # 2. TRAINING
        print(f"2️⃣  TRAINING (На спутниках - ПАРАЛЛЕЛЬНО)")
        metrics = []
        for sat in satellites:
            m = sat.train_on_satellite(epochs=epochs, lr=0.01)
            metrics.append(m)
            print(f"   ✓ Спутник {sat.sat_id}: Loss={m['loss']:.4f}, Acc={m['accuracy']:.4f}, Time={m['time']:.2f}s")
        print()
        
        # 3. AGGREGATE
        print(f"3️⃣  AGGREGATE (На Земле)")
        ground_station.aggregate_weights(satellites)
        print(f"   ✓ Веса успешно усреднены\n")
        
        # Stats
        avg_loss = np.mean([m['loss'] for m in metrics])
        avg_acc = np.mean([m['accuracy'] for m in metrics])
        
        print(f"📊 Статистика раунда {round_num + 1}:")
        print(f"   • Средняя Loss: {avg_loss:.4f}")
        print(f"   • Средняя Accuracy: {avg_acc:.4f}")
        print(f"   • Данные остались на спутниках ✓")
    
    return ground_station


if __name__ == "__main__":
    print("\n🛰️  === FedLEO v2.0: ОБУЧЕНИЕ ПОЛНОСТЬЮ НА СПУТНИКАХ ===\n")
    
    from tensorflow.keras.datasets import mnist
    print("📥 Загрузка MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    print(f"✓ Загружено: {x_train.shape[0]} примеров\n")
    
    def create_model():
        return keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(784,)),
            layers.Dense(10, activation='softmax')
        ])
    
    print("🛰️  Создание спутников...")
    satellites = []
    for i in range(4):
        start = i * 15000
        end = start + 15000
        sat = SatelliteV2(i, create_model(), x_train[start:end], y_train[start:end])
        satellites.append(sat)
    print()
    
    ground_station = GroundStationV2(create_model())
    ground_station = fedleo_v2_training(satellites, ground_station, num_rounds=2, epochs=1)
    
    print("\n" + "="*80)
    print("🧪 ТЕСТИРОВАНИЕ")
    print("="*80)
    
    ground_station.global_model.compile(
        optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )
    test_loss, test_acc = ground_station.global_model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%\n")
