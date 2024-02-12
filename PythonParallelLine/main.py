import cv2
import numpy as np

video_cap = cv2.VideoCapture(0)

while True:
    ret, frame = video_cap.read()
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # Reference fromhttps://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    enhanced_frame = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    edges = cv2.Canny(enhanced_frame, 100, 300)

    cv2.imshow('edges', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 70) # Reference from “https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html”
    if lines is None or len(lines) < 2:
        lines = []
    lines = sorted(lines, key=lambda x: x[0][1])

    angles = []
    parallel_lines = []
    for line_idx, line in enumerate(lines):
        match_flag = False
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        for tgt_rho, tgt_angle, tgt_line_idx in angles:
            if np.abs(tgt_angle - angle) < np.pi / 30 and tgt_line_idx != line_idx and np.abs(rho - tgt_rho) > 100:
                match_flag = True
                parallel_lines.append([line_idx, tgt_line_idx])
                break
        if not match_flag:
            angles.append([rho, angle, line_idx])
        else:
            break

    for line_pair in parallel_lines:
        for line_idx in line_pair:
            rho, theta = lines[line_idx][0] # Reference from“https://answers.opencv.org/question/2966/how-do-the-rho-and-theta-values-work-in-houghlines/”
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if len(parallel_lines) > 0:
        line1_idx, line2_idx = parallel_lines[0]
        rho1, theta1 = lines[line1_idx][0]
        rho2, theta2 = lines[line2_idx][0]

        x1 = rho1 * np.cos(theta1)
        y1 = rho1 * np.sin(theta1)
        x2 = rho2 * np.cos(theta2)
        y2 = rho2 * np.sin(theta2)

        mid_x = int((x1 + x2) // 2)
        mid_y = int((y1 + y2) // 2)
        mid_theta = (theta1 + theta2) / 2

        x1 = int(mid_x + 2000 * (-1 * np.sin(mid_theta)))
        y1 = int(mid_y + 2000 * (np.cos(mid_theta)))
        x2 = int(mid_x - 2000 * (-1 * np.sin(mid_theta)))
        y2 = int(mid_y - 2000 * (np.cos(mid_theta)))

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()