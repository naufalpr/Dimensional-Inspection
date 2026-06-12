import cv2
import numpy as np

REF_DIAMETER_CM = 2.72   # Koin 1000 (2,4 cm)
TARGET_P_CM = 5.2       
TARGET_L_CM = 2.2       
TOLERANSI_CM = 0.2      

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. PREPROCESSING
    h_f, w_f, _ = frame.shape
    roi = frame[50:h_f-50, 100:w_f-100].copy() 
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edged = cv2.Canny(blur, 40, 120)
    
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. DETEKSI KONTUR
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ppm = 0
    potential_targets = []

    # TAHAP A: MENCARI KOIN BERDASARKAN KEBULATAN/CIRCULARITY
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 500: continue

        # Hitung seberapa bulat objeknya
        peri = cv2.arcLength(c, True)
        circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0

        # Koin memiliki circularity > 0.7
        if circularity > 0.7 and 1500 < area < 10000:
            rect_k = cv2.minAreaRect(c)
            (x_k, y_k), (w_k, h_k), _ = rect_k
            ppm = max(w_k, h_k) / REF_DIAMETER_CM
            
            # Visual Koin
            box_k = np.array(cv2.boxPoints(rect_k), dtype="int")
            cv2.drawContours(roi, [box_k], -1, (255, 100, 0), 2)
            cv2.putText(roi, "REF KOIN", (int(x_k)-30, int(y_k)-20), 1, 0.8, (255, 100, 0), 1)
        else:
            # Jika tidak bulat, simpan sebagai kandidat komponen
            potential_targets.append(c)

    # TAHAP B: UKUR SEMUA KOMPONEN YANG DITEMUKAN
    count_ok = 0
    count_defect = 0

    if ppm > 0:
        for c in potential_targets:
            area = cv2.contourArea(c)
            if area < 1000: continue # Filter noise

            rect = cv2.minAreaRect(c)
            (x, y), (w, h), _ = rect
            
            p_cm = max(w, h) / ppm
            l_cm = min(w, h) / ppm

            # Filter agar hanya mendeteksi objek yang ukurannya mirip target (± 2cm)
            if (TARGET_P_CM - 2) < p_cm < (TARGET_P_CM + 2):
                is_ok = abs(p_cm - TARGET_P_CM) <= TOLERANSI_CM and abs(l_cm - TARGET_L_CM) <= TOLERANSI_CM
                
                if is_ok: count_ok += 1
                else: count_defect += 1
                
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                box = np.array(cv2.boxPoints(rect), dtype="int")
                
                # Gambar Box & Tulisan Ukuran di tiap objek
                cv2.drawContours(roi, [box], -1, color, 2)
                cv2.putText(roi, f"{p_cm:.1f}x{l_cm:.1f}cm", (int(x)-40, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. SUMMARY DASHBOARD
    cv2.rectangle(roi, (10, 10), (220, 90), (30, 30, 30), -1)
    cv2.putText(roi, f"PASS: {count_ok}", (20, 40), 1, 1.2, (0, 255, 0), 2)
    cv2.putText(roi, f"FAIL: {count_defect}", (20, 70), 1, 1.2, (0, 0, 255), 2)

    cv2.imshow("Multi-Object QC System", roi)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()