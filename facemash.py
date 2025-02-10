import cv2
import mediapipe as mp

# Inisialisasi Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Konfigurasi kamera
cap = cv2.VideoCapture(0)

# ID landmark untuk jari (urutan di Mediapipe)
finger_tips = [4, 8, 12, 16, 20]  # Ibu jari, telunjuk, jari tengah, manis, kelingking

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Ubah ke format RGB untuk Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=5),
                       mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3))

            # Hitung jari yang terangkat
            fingers_up = 0
            for i, tip in enumerate(finger_tips):
                if i == 0:  # Ibu jari memiliki logika berbeda
                    if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x:
                        fingers_up += 1
                else:  # Jari lainnya
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                        fingers_up += 1

            # Tampilkan angka jari yang terangkat
            cv2.putText(frame, f"Angka: {fingers_up}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking - Deteksi Angka", frame)

    # Tutup dengan tombol "Q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup program
hands.close()
cap.release()
cv2.destroyAllWindows()
