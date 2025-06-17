import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import BotSort
import torch
from pathlib import Path
import time
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, Tuple, List


class ObjectDetector:
    """
    Klasa odpowiedzialna za ładowanie modelu YOLO i detekcję obiektów.
    Odpowiada komponentowi ObjectDetector z diagramu UML.
    """

    def __init__(self, model_path: Path, device: str, half_precision: bool):
        self.device = device
        self.half_precision = half_precision
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path) -> YOLO:
        """Ładuje wstępnie wytrenowany model YOLO."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Plik wag YOLO nie został znaleziony: {model_path}")
        print(f"Ładowanie modelu YOLO z: {model_path}")
        return YOLO(model_path)

    def detect(self, frame: np.ndarray, conf: float, iou: float, inference_size: int) -> np.ndarray:
        """
        Przeprowadza detekcję obiektów na danej klatce.
        Zwraca detekcje wyłącznie dla klasy 'person' (ID 0).
        """
        results = self.model(
            frame,
            device=self.device,
            conf=conf,
            iou=iou,
            imgsz=inference_size,
            augment=False,
            half=self.half_precision,
            verbose=False,
            classes=[0]  # Detekcja tylko pieszych (klasa 0 w COCO)
        )
        return results[0].boxes.data.cpu().numpy()


class ObjectTracker:
    """
    Klasa odpowiedzialna za inicjalizację i aktualizację trackera BotSort.
    Odpowiada komponentowi ObjectTracker z diagramu UML.
    """

    def __init__(self, reid_weights_path: Path, device: str, half_precision: bool):
        self.tracker = self._setup_tracker(reid_weights_path, device, half_precision)

    def _setup_tracker(self, reid_weights_path: Path, device: str, half_precision: bool) -> BotSort:
        """Konfiguruje i zwraca instancję trackera BotSort."""
        if not reid_weights_path.is_file():
            raise FileNotFoundError(f"Plik wag ReID nie znaleziony: {reid_weights_path}")
        print("Konfiguracja trackera BotSort...")
        return BotSort(
            reid_weights=reid_weights_path,
            device=device,
            half=half_precision,
            track_high_thresh=0.5,
            track_low_thresh=0.3,
            new_track_thresh=0.8,
            match_thresh=0.85
        )

    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Aktualizuje stan trackera o nowe detekcje."""
        if len(detections) > 0:
            return self.tracker.update(detections, frame)
        return np.array([])


class Analytics:
    """
    Klasa odpowiedzialna za gromadzenie i przetwarzanie danych analitycznych.
    Odpowiada komponentowi Analytics z diagramu UML.
    """

    def __init__(self, heatmap_shape: Tuple[int, int], heatmap_rate: float, frame_window: int):
        self.track_history: Dict[int, deque] = {}
        self.total_people: set = set()
        self.heatmap: np.ndarray = np.zeros(heatmap_shape, dtype=np.float32)
        self.heatmap_accumulation_rate = heatmap_rate
        self.frame_window = frame_window
        self.avg_fps_list: List[float] = []
        print("Moduł analityczny zainicjalizowany.")

    def update_history(self, track_id: int, cx: int, cy: int):
        """Aktualizuje historię pozycji dla danego ID."""
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.frame_window)
        self.track_history[track_id].append((cx, cy, time.time()))
        self.total_people.add(track_id)

    def update_heatmap(self, cx: int, cy: int):
        """Inkrementuje wartość mapy ciepła w danej lokalizacji."""
        if 0 <= cy < self.heatmap.shape[0] and 0 <= cx < self.heatmap.shape[1]:
            self.heatmap[cy, cx] += self.heatmap_accumulation_rate

    def calculate_status(self, track_id: int, speed_threshold: float) -> str:
        """Oblicza prędkość i status (aktywny/pasywny) obiektu."""
        if len(self.track_history[track_id]) < 2:
            return "pasywny"

        speeds = []
        hist = list(self.track_history[track_id])
        for i in range(1, len(hist)):
            px, py, pt = hist[i - 1]
            nx, ny, nt = hist[i]
            dt = nt - pt
            if dt > 1e-6:
                dist = np.sqrt((nx - px) ** 2 + (ny - py) ** 2)
                speed = dist / dt
                if speed < 1000:  # Filtracja artefaktów prędkości
                    speeds.append(speed)

        if len(speeds) > 0 and np.median(speeds) >= speed_threshold:
            return "aktywny"
        return "pasywny"

    def export_results(self, output_dir: Path, last_frame: np.ndarray):
        """Zapisuje wszystkie wyniki analityczne do plików."""
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Rozpoczęto zapisywanie wyników...")

        # Zapis ostatniej klatki
        if last_frame is not None:
            last_frame_path = output_dir / 'last_frame_oop.jpg'
            cv2.imwrite(str(last_frame_path), last_frame)
            print(f"Ostatnia klatka zapisana jako: {last_frame_path}")

        # Zapis wykresu FPS
        if self.avg_fps_list:
            fps_plot_path = output_dir / 'fps_plot_oop.png'
            plt.figure(figsize=(10, 5))
            plt.plot(self.avg_fps_list, label='Średni FPS (kroczący)', color='orange')
            plt.xlabel('Numer klatki')
            plt.ylabel('FPS')
            plt.title('Średnia liczba klatek na sekundę (FPS) podczas przetwarzania')
            plt.legend()
            plt.grid(True)
            plt.savefig(str(fps_plot_path))
            plt.close()
            print(f"Wykres FPS zapisany jako: {fps_plot_path}")

        # Zapis heatmapy
        if np.sum(self.heatmap) > 0:
            heatmap_path = output_dir / 'heatmap_oop.jpg'

            max_val = np.percentile(self.heatmap, 99.9)
            heatmap_clipped = np.clip(self.heatmap, 0, max_val)
            heatmap_normalized = cv2.normalize(heatmap_clipped, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_blurred = cv2.GaussianBlur(heatmap_normalized, (15, 15), sigmaX=5, sigmaY=5)
            heatmap_colored = cv2.applyColorMap(heatmap_blurred.astype(np.uint8), cv2.COLORMAP_HOT)
            heatmap_colored = cv2.convertScaleAbs(heatmap_colored, alpha=1.5, beta=0)

            cv2.imwrite(str(heatmap_path), heatmap_colored)
            print(f"Mapa ciepła zapisana jako: {heatmap_path}")

            if last_frame is not None:
                heatmap_resized = cv2.resize(heatmap_colored, (last_frame.shape[1], last_frame.shape[0]))
                overlay = cv2.addWeighted(last_frame, 0.5, heatmap_resized, 0.7, 0)
                cv2.imwrite(str(output_dir / 'heatmap_overlay_oop.jpg'), overlay)


class VideoProcessor:
    """
    Główna klasa orkiestrująca cały proces przetwarzania wideo.
    Odpowiada komponentowi VideoProcess z diagramu UML.
    """

    def __init__(self, config: dict):
        self.config = config
        self._setup_device()

        # Inicjalizacja komponentów
        self.detector = ObjectDetector(
            model_path=Path(config['yolo_weights']),
            device=self.device,
            half_precision=self.half_precision
        )
        self.tracker = ObjectTracker(
            reid_weights_path=Path(config['reid_weights']),
            device=self.device,
            half_precision=self.half_precision
        )

        self.cap = cv2.VideoCapture(config['video_path'])
        if not self.cap.isOpened():
            raise IOError(f"Nie można otworzyć pliku wideo: {config['video_path']}")

        self.target_resolution = config['target_resolution']
        self.mask = self._load_mask(config['use_mask'], config.get('mask_path'))

        heatmap_shape = (self.target_resolution[1], self.target_resolution[0])
        self.analytics = Analytics(
            heatmap_shape=heatmap_shape,
            heatmap_rate=config['heatmap_accumulation_rate'],
            frame_window=config['frame_window']
        )

        self.fps_list = deque(maxlen=100)
        self.last_frame = None

    def _setup_device(self):
        """Wybiera odpowiednie urządzenie do obliczeń (CUDA, MPS, CPU)."""
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.half_precision = True
        else:
            self.device = 'cpu'
            self.half_precision = False
        print(
            f"Używane urządzenie: {self.device}, Precyzja połowiczna: {'Włączona' if self.half_precision else 'Wyłączona'}")

    def _load_mask(self, use_mask: bool, mask_path: str) -> np.ndarray | None:
        """Ładuje i przetwarza maskę ROI."""
        if not use_mask or not mask_path:
            return None
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError("Nie można wczytać pliku maski.")
            mask = cv2.resize(mask, self.target_resolution)
            _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
            print(f"Maska '{mask_path}' wczytana pomyślnie.")
            return mask
        except Exception as e:
            print(f"BŁĄD wczytywania maski: {e}. Kontynuacja bez maski.")
            return None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Skaluje klatkę do docelowej rozdzielczości."""
        return cv2.resize(frame, self.target_resolution)

    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """Nakłada maskę ROI na klatkę, jeśli jest dostępna."""
        if self.mask is not None:
            return cv2.bitwise_and(frame, frame, mask=self.mask)
        return frame

    def visualize(self, frame: np.ndarray, tracked_objects: np.ndarray, stats: dict) -> np.ndarray:
        """Rysuje wyniki analizy na klatce."""
        draw_frame = frame.copy()
        track_overlay = np.zeros_like(draw_frame, dtype=np.uint8)

        # Rysowanie śladów
        if self.config['draw_tracks']:
            for track_id, history in self.analytics.track_history.items():
                points = np.array([p[:2] for p in history], dtype=np.int32).reshape((-1, 1, 2))
                if len(points) > 1:
                    cv2.polylines(track_overlay, [points], False, (255, 255, 0), 2)

        draw_frame = cv2.addWeighted(draw_frame, 1, track_overlay, 0.7, 0)

        # Rysowanie ramek i ID
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj[:5])
            status = stats['statuses'].get(track_id, "pasywny")
            color = (0, 255, 0) if status == "aktywny" else (0, 0, 255)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Rysowanie statystyk
        y_offset = 30
        cv2.putText(draw_frame, f"FPS: {stats['current_fps']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Avg FPS: {stats['avg_fps']:.2f}", (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Current: {stats['current_people']}", (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Total Unique: {stats['total_unique']}", (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Active: {stats['active_count']}", (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(draw_frame, f"Passive: {stats['passive_count']}", (10, y_offset + 150), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        return draw_frame

    def run(self):
        """Główna pętla przetwarzająca wideo."""
        frame_count = 0
        print("Rozpoczęcie przetwarzania wideo...")

        while self.cap.isOpened():
            start_frame_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("Koniec strumienia wideo.")
                break

            frame_count += 1

            # 1. Przygotowanie klatki
            preprocessed_frame = self._preprocess(frame)

            # 2. Nałożenie maski
            masked_frame = self.apply_roi_mask(preprocessed_frame)

            # 3. Detekcja
            detections = self.detector.detect(
                masked_frame,
                conf=self.config['conf_threshold'],
                iou=self.config['iou_threshold'],
                inference_size=self.config['inference_size']
            )

            # 4. Śledzenie
            tracked_objects = self.tracker.update(detections, masked_frame)

            # 5. Analiza
            active_count = 0
            statuses = {}
            for obj in tracked_objects:
                if len(obj) < 5: continue
                x1, y1, x2, y2, track_id = map(int, obj[:5])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                self.analytics.update_history(track_id, cx, cy)
                self.analytics.update_heatmap(cx, cy)
                status = self.analytics.calculate_status(track_id, self.config['speed_threshold'])
                statuses[track_id] = status
                if status == "aktywny":
                    active_count += 1

            # Obliczenie FPS
            frame_time = time.time() - start_frame_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            self.fps_list.append(current_fps)
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            self.analytics.avg_fps_list.append(avg_fps)

            # Przygotowanie statystyk do wizualizacji
            stats = {
                "current_fps": current_fps,
                "avg_fps": avg_fps,
                "current_people": len(tracked_objects),
                "total_unique": len(self.analytics.total_people),
                "active_count": active_count,
                "passive_count": len(tracked_objects) - active_count,
                "statuses": statuses
            }

            # 6. Wizualizacja
            display_frame = self.visualize(masked_frame, tracked_objects, stats)
            self.last_frame = display_frame.copy()

            cv2.imshow('Tracking (Object-Oriented)', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Przerwano przez użytkownika.")
                break

        # Zakończenie i zapis wyników
        self.cap.release()
        cv2.destroyAllWindows()
        print("Zwolniono zasoby wideo.")

        self.analytics.export_results(
            output_dir=Path(self.config['output_dir']),
            last_frame=self.last_frame
        )
        print("Zakończono działanie skryptu.")


if __name__ == "__main__":
    # Parametry konfiguracyjne zebrane w jednym miejscu
    CONFIG = {
        "video_path": "video02.mp4",
        "output_dir": "picture",
        "yolo_weights": "YoloWeights/yolo11x.pt",
        "reid_weights": "resnet50_fc512_msmt17.pt",
        "mask_path": "picture/maska_dublin.png",

        "use_mask": input("Czy chcesz użyć maski? (tak/nie): ").strip().lower() == 'tak',
        "draw_tracks": input("Czy chcesz wizualizować trasy? (tak/nie): ").strip().lower() == 'tak',

        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "inference_size": 1600,
        "target_resolution": (1280, 720),

        "speed_threshold": 15,
        "frame_window": 150,
        "heatmap_accumulation_rate": 30.0,
    }

    try:
        processor = VideoProcessor(config=CONFIG)
        processor.run()
    except (FileNotFoundError, IOError, Exception) as e:
        print(f"FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()