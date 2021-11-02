import cv2
import glob, math, os
import numpy as np
import time

resW = 3840
resH = 2160

outResW = 1280
outResH = 720
faceSize = 512

bDebug = False

def main():
    wName = "input"

    cv2.namedWindow(wName)
    cv2.setWindowProperty(wName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    files = glob.glob("input/**/*.jpg", recursive=True)

    for i in files:
 
        src = cv2.imread(i)
        
        if np.shape(src) == ():
            print(i + " is skipped")
            continue

        param = {
            "cnt" : 0,
            "points": [],
            "isFinished": False,
        }

        print("Select eyes: " + i)
        print(param)
        def mouse_click(event,x,y,flags,param):
            if param["isFinished"] is True:
                return 

            if event == cv2.EVENT_LBUTTONUP:
                param["cnt"] += 1
                param["points"].append(np.array([x,y]))
                print(f'{param["cnt"]}:{x},{y}')
                if param["cnt"] == 4:
                    param["isFinished"] = True

        cv2.setMouseCallback(wName, mouse_click, param)
        h,w, _ = np.shape(src)
        dispH = resH
        dispW = w * resH / h
        disp_src = cv2.resize(src, (int(dispW), int(dispH)))
    
        while param["isFinished"] is False:
            cv2.imshow("input", disp_src)
            key = cv2.waitKey(16)
            if key == 32:
                # press space
                break
            if key == ord('q'):
                return

        if param["isFinished"] is False:
            continue

        # selected 4points
        e0_, e1_, m0_, m1_ = param["points"]
        e0 = np.array([e0_[0], e0_[1]]) * h / resH
        e1 = np.array([e1_[0], e1_[1]]) * h / resH
        m0 = np.array([m0_[0], m0_[1]]) * h / resH
        m1 = np.array([m1_[0], m1_[1]]) * h / resH

        x_ = e1 - e0
        y_ = 0.5 * (e0 + e1) - 0.5 * (m0 + m1)
        c = 0.5 * (e0 + e1) - 0.1 * y_
        s = max(4.0 * np.linalg.norm(x_, ord=2), 3.6 * np.linalg.norm(y_, ord=2))
        x = x_ - np.array([y_[1], -y_[0]])
        x = x / np.linalg.norm(x, ord=2)
        y = np.array([-x[1], x[0]])

        print(x, y, s)
 
        zero = c - 0.5 * s * x - 0.5 * s * y 
        print(zero)


        if bDebug:
            vec1 = zero + s * x
            vec2 = zero + s * y
            cv2.line(src, (int(zero[0]), int(zero[1])), (int(vec1[0]), int(vec1[1])) , (255,0,0), 4)
            cv2.line(src, (int(zero[0]), int(zero[1])), (int(vec2[0]), int(vec2[1])) , (255,0,0), 4)
            cv2.imshow(wName, src)
            cv2.waitKey(0)


        M = cv2.getAffineTransform(
            np.array([zero, zero + s * x, zero + s * y], dtype=np.float32),
            np.array([[0.5 * (outResW - faceSize), 0.5 * (outResH - faceSize)],
            [0.5 * (outResW + faceSize), 0.5 * (outResH - faceSize)],
            [0.5 * (outResW - faceSize), 0.5 * (outResH + faceSize)]], dtype=np.float32)
        )

        out_img = cv2.warpAffine(src, M, (outResW, outResH))

        filedir, filename = os.path.split(i)
        print(filedir)
        outputdir = filedir.replace('input', 'output')
        os.makedirs(outputdir, exist_ok=True)
        cv2.imwrite(os.path.join(outputdir, filename), out_img)

        cv2.imshow(wName, out_img)
        cv2.waitKey(0)



if __name__ == "__main__":
    main()