import cv2

# Replace 'rtsp://username:password@camera_ip_address:port' with your camera's RTSP URL
rtsp_url = 'rtsp://162it171:162it171@10.10.133.1:554'

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('IP Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
