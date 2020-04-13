import os

import cv2


def bb_iou(box1, box2):
    """
    computes intersection over union (iou) between 2 bounding boxes.
    :param box1: prediction box.
    :param box2: gt box.
    :return: iou.
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
    # object is lost, no box
    if x_right == 0 or y_bottom == 0:
        return 0.0
    overlap_area = (x_right - x_left) * (y_bottom - y_top)  # intersection
    bb1_area = box1[2] * box1[3]
    bb2_area = box2[2] * box2[3]
    combined_area = bb1_area + bb2_area - overlap_area  # union
    iou = overlap_area / float(combined_area)
    return iou


def evaluate(predictions, ground_truth, iou_cutoff=0.5):
    """
    evaluation method.
    :param predictions: dict of the predictions as tuples.
    :param ground_truth: dict of the gt as tuples.
    :param iou_cutoff: value at which an iou is considered as true positives.
    :return: accuracy and robustness metrics.
    accuracy = ratio of the number of times the object was correctly tracked across all frames.
    robustness = precision of the tracking when the object was correctly tracked.
    """
    assert len(predictions) == len(ground_truth)
    tp = 0
    mean_iou = 0.0
    for i in range(len(ground_truth)):
        prediction = predictions[i]
        gt = ground_truth[i]
        iou = bb_iou(prediction, gt)

        if iou >= iou_cutoff:
            tp += 1
            mean_iou += iou
    return float(tp) / len(ground_truth), float(mean_iou) / float(tp)


def get_frames(folder):
    """
    get the full path to all the frames in the video sequence.
    :param folder: path to the folder containing the frames of the video sequence.
    :return: list of the name of all the frames.
    """
    names = os.listdir(folder)
    frames = [os.path.join(folder, n) for n in names if n.endswith('.jpg')]
    frames.sort()
    return frames


def init_tracker(gt):
    """
    get the object location in the first frame.
    :param gt: box location for each frame (output of read_ground_truth).
    :return: location of the object in the first frame.
    """
    return gt[0]


def read_ground_truth(path):
    """
    reads ground-truth and returns it as a numpy array.
    :param path: path to groundtruth.txt file.
    :return: dict of the 4 the coordinates for top-left corner and width/height as tuple.
    """
    ground_truth = {}
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            x, y, width, height = line.split(',')
            x = int(float(x))
            y = int(float(y))
            width = int(float(width))
            height = int(float(height))
            ground_truth[i] = (x, y, width, height)
    return ground_truth


def test_ground_truth(folder, gt):
    """
    use this function to see the ground-truth boxes on the sequence.
    :param folder: path to the folder containing the frames of the video sequence.
    :param gt: box location for each frame (output of read_ground_truth).
    :return: void
    """
    frames = get_frames(folder)

    for i, frame in enumerate(frames):
        box = gt[i]
        frame = cv2.imread(frame, cv2.IMREAD_COLOR)
        cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color=(0, 0, 255))
        cv2.imshow('sequence', frame)
        cv2.waitKey(delay=24)


def track(folder, gt):
    """
    code for your tracker.
    :param folder: path to the folder containing the frames of the video sequence.
    :param gt: box location for each frame (output of read_ground_truth).
    :return: dict with an entry for each frame being a tuple (x, y, width, height)
    """
    # TODO: code for tracking an object in a given sequence.
    
    #Chargement des images de la séquence vidéo testée
    imagePaths = get_frames(folder)
    frame = cv2.imread(imagePaths[0])
    (H, W) = frame.shape[:2]
    
    #Création du tracker CSRT
    tracker = tracker = cv2.TrackerCSRT_create()
    
    #Coordonnées de l'objet d'intérêt dans la première image
    initBB = gt[0]
    tracker.init(frame, initBB)
    
    #Pour l'estimation du temps de calcul
    fps = FPS().start()
    
    #Dictionnaire pour stocker les boîtes englobantes calculées
    computedBoxes = {}
    computedBoxes[0] = initBB
    
    key = "a"
    for i in range(1, len(imagePaths)): #On lit toute la vidéo
        if key == ord("q"): #Arrêter l'exécution en appuyant sur q
            break
        
        frame = cv2.imread(imagePaths[i])
        
        if frame is None:
            break
        
        #Suivi de l'objet dans l'image courante
        (success, box) = tracker.update(frame)
        
        if success:
            (x, y, w, h) = [int(v) for v in box] 
            computedBoxes[i] = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            computedBoxes[i] = computedBoxes[i-1]
            
        fps.update()
        fps.stop()
        
        #Affichage des statistiques
        info = [("Tracker", "csrt"), ("Success", "Yes" if success else "No"), ("FPS", "{:.2f}".format(fps.fps()))]
        
        for (j, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((j * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
            
    cv2.destroyAllWindows()
    
    return(computedBoxes)
    
    


def main():
    frames_folder = "dataset/bolt/frames"
    path_gt = "dataset/bolt/groundtruth.txt"
    gt = read_ground_truth(path_gt)
    # test_ground_truth(frames_folder, gt)
    predictions = track(frames_folder, gt)
    accuracy, robustness = evaluate(predictions, gt)
    print(f'accuracy = {accuracy}, robustness = {robustness}')


if __name__ == '__main__':
    main()
